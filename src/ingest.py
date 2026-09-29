"""Knowledge-base ingestion: S3 -> chunk -> embed -> pgvector.

Run on demand (locally, or as a one-off job), not as an always-on service:

    python -m src.ingest              # sync from S3, embed only new/changed documents
    python -m src.ingest --dry-run    # show plan + estimated cost; no OpenAI calls, no document changes
    python -m src.ingest --local      # skip S3, ingest the files already in data/raw
    python -m src.ingest --force      # re-embed every document
    python -m src.ingest --prune      # also delete documents that are no longer in the source

Unchanged documents (same file hash as what is stored) are skipped, so a routine
run costs nothing in embedding fees unless a document was added or edited.
"""
import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .config import settings
from .logger import setup_logging
from .chunker import DocumentChunker
from .embedder import EmbeddingService
from .vectorstore import PostgreSQLVectorStore

RAW_DIR = os.path.join("data", "raw")


def format_cost(usd: float) -> str:
    return f"${usd:.4f}" if usd >= 0.0001 or usd == 0 else "<$0.0001"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# USD per 1M tokens, for the dry-run estimate only
EMBEDDING_PRICE_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


@dataclass
class IngestPlan:
    source_names: Set[str]                                   # every document the source says should exist
    files: Dict[str, str]                                    # name -> local path, for documents available on disk
    new: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    orphans: List[str] = field(default_factory=list)         # stored, but no longer in the source

    @property
    def to_embed(self) -> List[str]:
        return self.new + self.changed


async def collect_source(use_s3: bool) -> tuple:
    """Return (source_names, files). Source names come from the S3 listing when syncing,
    so a failed download never makes a live document look deleted."""
    if use_s3:
        from .ingestion import S3IngestionService  # boto3 is only needed for S3 syncs

        s3 = S3IngestionService()
        objects = await s3.list_s3_objects()
        source_names = {s3._sanitize_filename(os.path.basename(o["key"])) for o in objects}
        await s3.sync_bucket_files()
    else:
        source_names = set()
        if os.path.isdir(RAW_DIR):
            source_names = {
                name for name in os.listdir(RAW_DIR)
                if os.path.splitext(name.lower())[1] in SUPPORTED_EXTENSIONS
            }

    files = {
        name: os.path.join(RAW_DIR, name)
        for name in sorted(source_names)
        if os.path.isfile(os.path.join(RAW_DIR, name))
    }
    return source_names, files


def build_plan(source_names: Set[str], files: Dict[str, str], stored: Dict[str, set],
               chunker: DocumentChunker, force: bool) -> IngestPlan:
    plan = IngestPlan(source_names=source_names, files=files)
    for name, path in files.items():
        file_hash = chunker._calculate_file_hash(path)
        if name not in stored:
            plan.new.append(name)
        elif force or stored[name] != {file_hash}:
            # Multiple stored hashes also count as changed: it cleans up stale chunks
            plan.changed.append(name)
        else:
            plan.unchanged.append(name)
    plan.orphans = sorted(set(stored) - source_names)
    return plan


def print_plan(plan: IngestPlan, prune: bool) -> None:
    missing = sorted(plan.source_names - set(plan.files))
    print(f"Source documents: {len(plan.source_names)} | new: {len(plan.new)} | "
          f"changed: {len(plan.changed)} | unchanged (skipped): {len(plan.unchanged)}")
    for label, names in (("NEW", plan.new), ("CHANGED", plan.changed)):
        for name in names:
            print(f"  {label:8} {name}")
    for name in missing:
        print(f"  MISSING  {name} (listed in source but not downloaded; left untouched)")
    for name in plan.orphans:
        action = "will delete" if prune else "kept; use --prune to delete"
        print(f"  ORPHAN   {name} (not in source; {action})")


async def run(args: argparse.Namespace) -> int:
    chunker = DocumentChunker()
    store = PostgreSQLVectorStore()
    embedder = None
    failures = 0

    try:
        source_names, files = await collect_source(use_s3=not args.local)
        stored = await store.get_document_hashes()
        plan = build_plan(source_names, files, stored, chunker, args.force)
        print_plan(plan, args.prune)

        if not source_names:
            # Guard against wiping the knowledge base because of an empty/misconfigured source
            print("No source documents found; refusing to continue.")
            return 1

        if args.dry_run:
            total_chars = 0
            for name in plan.to_embed:
                chunks = await chunker.process_document(plan.files[name])
                total_chars += sum(c.char_count for c in chunks)
            est_tokens = total_chars // 4
            price = EMBEDDING_PRICE_PER_MILLION.get(settings.embedding_model, 0.10)
            print(f"Dry run: would embed ~{est_tokens:,} tokens "
                  f"(~{format_cost(est_tokens / 1_000_000 * price)} with {settings.embedding_model}). No changes made.")
            return 0

        total_cost = 0.0
        if plan.to_embed:
            embedder = EmbeddingService()

        for name in plan.to_embed:
            try:
                chunks = await chunker.process_document(plan.files[name])
                if not chunks:
                    print(f"  SKIP     {name}: no extractable text (existing chunks left untouched)")
                    failures += 1
                    continue
                embeddings = await embedder.process_document_chunks(chunks)
                by_id = {e.chunk_id: e for e in embeddings}
                pairs = [(c, by_id[c.chunk_id].embedding) for c in chunks]
                deleted, inserted = await store.replace_document_chunks(name, pairs)
                total_cost += sum(e.cost_estimate or 0 for e in embeddings)
                print(f"  DONE     {name}: {inserted} chunks stored, {deleted} old chunks replaced")
            except Exception as e:
                failures += 1
                print(f"  FAILED   {name}: {e}")

        if args.prune and plan.orphans:
            deleted = await store.delete_documents_by_name(plan.orphans)
            print(f"Pruned {len(plan.orphans)} documents ({deleted} chunks)")

        if plan.to_embed or (args.prune and plan.orphans):
            had_ivfflat = store._ivfflat_enabled
            if await store.refresh_search_indexes() and not had_ivfflat:
                print("Note: an ivfflat index was created; restart the API so it starts using it.")

        print(f"Finished: {len(plan.to_embed) - failures} embedded, {failures} failed, "
              f"embedding cost ~{format_cost(total_cost)}")
        return 1 if failures else 0

    finally:
        if embedder:
            await embedder.close()
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the IUFP knowledge base.")
    parser.add_argument("--local", action="store_true", help="skip S3 sync; ingest files already in data/raw")
    parser.add_argument("--dry-run", action="store_true", help="show plan and estimated cost without changing anything")
    parser.add_argument("--force", action="store_true", help="re-embed all documents even if unchanged")
    parser.add_argument("--prune", action="store_true", help="delete stored documents that are no longer in the source")
    parser.add_argument("--verbose", action="store_true", help="show detailed structured logs")
    args = parser.parse_args()

    if not args.verbose:
        settings.log_level = "WARNING"
    setup_logging()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

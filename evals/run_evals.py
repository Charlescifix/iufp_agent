"""Run the chatbot against evals/cases.json and report which cases pass.

    python evals/run_evals.py [--only ID ...] [--no-judge] [--out results.json]

Uses the real database and OpenAI API, as the live /chat endpoint does
(retrieval + generate_response), without storing chat messages.
"""
import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import structlog
from openai import AsyncOpenAI

from src.chat_api import ChatService, is_no_answer
from src.config import settings
from src.retriever import HybridRetriever, RetrievalConfig

# App logs go to stderr at WARNING so the report on stdout stays readable
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
                    logger_factory=structlog.PrintLoggerFactory(sys.stderr))
logging.getLogger().setLevel(logging.WARNING)

OFF_TOPIC = re.compile(r"\bonly (help|assist)\b", re.IGNORECASE)
CONCURRENCY = 5

JUDGE_PROMPT = """You grade one reply from a university-admissions chatbot against a requirement.
Judge only the requirement given. Ignore style, length and extra caveats unless the requirement mentions them.
Answer in JSON: {"pass": true or false, "reason": "<one short sentence>"}"""


def normalise(text: str) -> str:
    text = text.replace("**", "").replace("’", "'")
    return re.sub(r"[‒-―]", "-", text).lower()


def check_rules(case: dict, reply: str) -> list:
    """Deterministic checks; returns the reasons the case failed."""
    failures = []
    expect = case["expect"]
    body = normalise(reply)
    no_answer = is_no_answer(reply)
    off_topic = bool(OFF_TOPIC.search(reply))

    if expect == "answer" and no_answer:
        failures.append("gave the no-answer reply")
    if expect == "answer" and off_topic:
        failures.append("refused as off-topic")
    if expect == "no_answer" and not no_answer:
        failures.append("expected the no-answer reply")
    if expect == "off_topic" and not off_topic:
        failures.append("expected an off-topic refusal")
    if expect == "identity" and ("assistant" not in body or no_answer or off_topic):
        failures.append("expected the bot to say it is IUFP's assistant")
    if expect == "greeting" and (no_answer or off_topic or len(reply.split()) > 40):
        failures.append("expected a short greeting reply")

    for term in case.get("include", []):
        if normalise(term) not in body:
            failures.append(f"missing {term!r}")
    any_terms = case.get("include_any", [])
    if any_terms and not any(normalise(t) in body for t in any_terms):
        failures.append(f"missing any of {any_terms}")
    for term in case.get("exclude", []):
        if normalise(term) in body:
            failures.append(f"contains forbidden {term!r}")
    return failures


async def judge(client: AsyncOpenAI, case: dict, reply: str) -> list:
    response = await client.chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Question: {case['question']}\n\nReply: {reply}\n\n"
                                        f"Requirement: {case['judge']}"},
        ],
    )
    verdict = json.loads(response.choices[0].message.content)
    return [] if verdict.get("pass") else [f"judge: {verdict.get('reason', 'failed')}"]


async def run_case(case, retriever, service, client, use_judge, semaphore) -> dict:
    async with semaphore:
        history = []
        for user_turn, bot_turn in case.get("history", []):
            history += [{"role": "user", "content": user_turn}, {"role": "assistant", "content": bot_turn}]
        try:
            chunks = await retriever.search(case["question"],
                                            RetrievalConfig(max_results=settings.max_retrieval_results))
            reply = await service.generate_response(case["question"], chunks, history)
        except Exception as e:
            return {**case, "reply": "", "chunks": 0, "failures": [f"error: {e}"]}

        failures = check_rules(case, reply)
        if use_judge and case.get("judge") and not failures:
            failures = await judge(client, case, reply)
        return {**case, "reply": reply, "chunks": len(chunks), "failures": failures}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these case ids")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (rule checks only)")
    parser.add_argument("--out", help="write every case, reply and verdict to this JSON file")
    args = parser.parse_args()

    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    retriever = HybridRetriever()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    service = ChatService(retriever.vector_store, retriever, client)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    results = await asyncio.gather(*(run_case(c, retriever, service, client, not args.no_judge, semaphore)
                                     for c in cases))

    by_category = defaultdict(lambda: [0, 0])
    for r in results:
        passed = not r["failures"]
        by_category[r["category"]][0] += passed
        by_category[r["category"]][1] += 1
        print(f"{'PASS' if passed else 'FAIL'}  {r['id']:<26} [{r['chunks']} chunks]")
        if not passed:
            print(f"      Q: {r['question']}")
            print(f"      A: {' '.join(r['reply'].split())[:300]}")
            for failure in r["failures"]:
                print(f"      - {failure}")

    total_passed = sum(p for p, _ in by_category.values())
    print("\nBy category:")
    for category, (p, n) in by_category.items():
        print(f"  {category:<14} {p}/{n}")
    print(f"\nTOTAL {total_passed}/{len(results)} passed")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if total_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

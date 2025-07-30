import re
import json
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pdfplumber
from datetime import datetime
import hashlib

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event


@dataclass
class DocumentChunk:
    """Data class for document chunks with metadata"""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    char_count: int
    word_count: int
    source_hash: str
    created_at: str
    section_title: Optional[str] = None
    metadata: Optional[Dict] = None


class ChunkingSecurityError(Exception):
    pass


class DocumentChunker:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.max_chunk_size = settings.max_chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
    def _validate_file_path(self, file_path: str) -> None:
        """Validate file path for security"""
        log_function_call(self.logger, "_validate_file_path", file_path=file_path[:50])
        
        # Ensure file exists
        if not os.path.exists(file_path):
            error = ChunkingSecurityError(f"File does not exist: {file_path}")
            log_function_result(self.logger, "_validate_file_path", error=error)
            raise error
        
        # Ensure file is within allowed directory
        abs_path = os.path.abspath(file_path)
        allowed_base = os.path.abspath("data/raw")
        if not abs_path.startswith(allowed_base):
            error = ChunkingSecurityError(f"File outside allowed directory: {file_path}")
            log_security_event(
                "path_traversal_attempt",
                {"file_path": file_path, "allowed_base": allowed_base},
                "ERROR"
            )
            log_function_result(self.logger, "_validate_file_path", error=error)
            raise error
        
        # Validate file extension
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        _, ext = os.path.splitext(file_path.lower())
        if ext not in allowed_extensions:
            error = ChunkingSecurityError(f"File extension {ext} not allowed")
            log_security_event(
                "invalid_file_extension",
                {"file_path": file_path, "extension": ext},
                "WARNING"
            )
            log_function_result(self.logger, "_validate_file_path", error=error)
            raise error
        
        # Check file size (prevent processing extremely large files)
        file_size = os.path.getsize(file_path)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            error = ChunkingSecurityError(f"File too large: {file_size} bytes (max: {max_size})")
            log_security_event(
                "file_size_exceeded",
                {"file_path": file_path, "size": file_size, "max_size": max_size},
                "WARNING"
            )
            log_function_result(self.logger, "_validate_file_path", error=error)
            raise error
        
        log_function_result(self.logger, "_validate_file_path")
    
    def _extract_text_from_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF with metadata using pdfplumber"""
        log_function_call(self.logger, "_extract_text_from_pdf", file_path=file_path[:50])
        
        return self._extract_with_pdfplumber(file_path)
    
    def _extract_with_pdfplumber(self, file_path: str) -> Tuple[str, Dict]:
        """Fallback extraction using pdfplumber"""
        log_function_call(self.logger, "_extract_with_pdfplumber", file_path=file_path[:50])
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                metadata = {
                    'total_pages': len(pdf.pages),
                    'extraction_method': 'pdfplumber',
                    'page_texts': {}
                }
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    
                    if page_text:
                        page_text = self._clean_extracted_text(page_text)
                        text_parts.append(page_text)
                        metadata['page_texts'][page_num + 1] = len(page_text)
                
                full_text = '\n\n'.join(text_parts)
                
                self.logger.info(
                    "PDF text extracted with pdfplumber",
                    file_path=file_path,
                    pages=len(pdf.pages),
                    text_length=len(full_text),
                    method="pdfplumber"
                )
                
                log_function_result(self.logger, "_extract_with_pdfplumber", result=f"Extracted {len(full_text)} chars")
                return full_text, metadata
                
        except Exception as e:
            error = ChunkingSecurityError(f"Failed to extract text from PDF: {str(e)}")
            log_function_result(self.logger, "_extract_with_pdfplumber", error=error)
            raise error
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and sanitize extracted text"""
        if not text:
            return ""
        
        # Remove null bytes and other control characters that could cause issues
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessively long repeated characters (potential DoS)
        text = re.sub(r'(.)\1{50,}', r'\1\1\1', text)
        
        # Limit total text length to prevent memory issues
        max_text_length = 10 * 1024 * 1024  # 10MB of text
        if len(text) > max_text_length:
            text = text[:max_text_length]
            self.logger.warning("Text truncated due to length", original_length=len(text), max_length=max_text_length)
        
        return text.strip()
    
    def _extract_text_from_txt(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from plain text file"""
        log_function_call(self.logger, "_extract_text_from_txt", file_path=file_path[:50])
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            text = self._clean_extracted_text(text)
            metadata = {
                'extraction_method': 'plain_text',
                'encoding': 'utf-8'
            }
            
            log_function_result(self.logger, "_extract_text_from_txt", result=f"Extracted {len(text)} chars")
            return text, metadata
            
        except Exception as e:
            error = ChunkingSecurityError(f"Failed to extract text from file: {str(e)}")
            log_function_result(self.logger, "_extract_text_from_txt", error=error)
            raise error
    
    def _generate_document_id(self, file_path: str, file_hash: str) -> str:
        """Generate unique document ID"""
        filename = os.path.basename(file_path)
        doc_string = f"{filename}:{file_hash}"
        return hashlib.md5(doc_string.encode()).hexdigest()
    
    def _split_text_into_chunks(self, text: str, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        log_function_call(self.logger, "_split_text_into_chunks", text_length=len(text))
        
        if overlap is None:
            overlap = self.chunk_overlap
        
        # Validate chunk parameters
        if self.max_chunk_size <= 0 or overlap < 0:
            error = ChunkingSecurityError("Invalid chunk parameters")
            log_function_result(self.logger, "_split_text_into_chunks", error=error)
            raise error
        
        if overlap >= self.max_chunk_size:
            overlap = self.max_chunk_size // 2
            self.logger.warning("Overlap reduced to prevent infinite chunks", new_overlap=overlap)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings in the last 200 characters
                last_part = chunk_text[-200:]
                sentence_ends = ['.', '!', '?', '\n\n']
                
                best_break = -1
                for sent_end in sentence_ends:
                    pos = last_part.rfind(sent_end)
                    if pos > best_break:
                        best_break = pos
                
                if best_break > 100:  # Only break if we found a good spot
                    chunk_text = chunk_text[:-(200 - best_break - 1)]
            
            chunks.append(chunk_text.strip())
            
            # Move start position with overlap
            if end >= len(text):
                break
            start = end - overlap
        
        self.logger.debug("Text split into chunks", total_chunks=len(chunks), avg_chunk_size=len(text) // len(chunks) if chunks else 0)
        log_function_result(self.logger, "_split_text_into_chunks", result=f"{len(chunks)} chunks")
        return chunks
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of source file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document into chunks with full security validation"""
        log_function_call(self.logger, "process_document", file_path=file_path[:50])
        
        try:
            # Security validation
            self._validate_file_path(file_path)
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)
            document_id = self._generate_document_id(file_path, file_hash)
            document_name = os.path.basename(file_path)
            
            # Extract text based on file type
            _, ext = os.path.splitext(file_path.lower())
            if ext == '.pdf':
                text, extraction_metadata = self._extract_text_from_pdf(file_path)
            elif ext in ['.txt']:
                text, extraction_metadata = self._extract_text_from_txt(file_path)
            else:
                error = ChunkingSecurityError(f"Unsupported file type: {ext}")
                log_function_result(self.logger, "process_document", error=error)
                raise error
            
            if not text.strip():
                self.logger.warning("No text extracted from document", file_path=file_path)
                return []
            
            # Split into chunks
            chunk_texts = self._split_text_into_chunks(text)
            
            # Create chunk objects
            chunks = []
            created_at = datetime.utcnow().isoformat()
            
            for i, chunk_text in enumerate(chunk_texts):
                chunk_id = hashlib.md5(f"{document_id}:chunk:{i}".encode()).hexdigest()
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=None,  # Could be enhanced to track page numbers
                    chunk_index=i,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    source_hash=file_hash,
                    created_at=created_at,
                    metadata={
                        'extraction_metadata': extraction_metadata,
                        'chunk_overlap': self.chunk_overlap,
                        'max_chunk_size': self.max_chunk_size
                    }
                )
                chunks.append(chunk)
            
            self.logger.info(
                "Document processed successfully",
                file_path=file_path,
                document_id=document_id,
                total_chunks=len(chunks),
                total_chars=len(text),
                avg_chunk_size=sum(c.char_count for c in chunks) // len(chunks) if chunks else 0
            )
            
            log_function_result(self.logger, "process_document", result=f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            log_function_result(self.logger, "process_document", error=e)
            raise
    
    async def save_chunks_to_json(self, chunks: List[DocumentChunk], output_dir: str = "data/chunks") -> str:
        """Save chunks to JSON file with security validation"""
        log_function_call(self.logger, "save_chunks_to_json", chunk_count=len(chunks), output_dir=output_dir[:50])
        
        if not chunks:
            self.logger.warning("No chunks to save")
            return ""
        
        try:
            # Validate output directory
            abs_output_dir = os.path.abspath(output_dir)
            allowed_base = os.path.abspath("data")
            if not abs_output_dir.startswith(allowed_base):
                error = ChunkingSecurityError(f"Output directory outside allowed path: {output_dir}")
                log_security_event(
                    "path_traversal_attempt",
                    {"output_dir": output_dir, "allowed_base": allowed_base},
                    "ERROR"
                )
                log_function_result(self.logger, "save_chunks_to_json", error=error)
                raise error
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename based on document
            document_id = chunks[0].document_id
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"chunks_{document_id}_{timestamp}.json"
            output_path = os.path.join(output_dir, filename)
            
            # Convert chunks to dict format
            chunks_data = {
                'metadata': {
                    'document_id': document_id,
                    'document_name': chunks[0].document_name,
                    'total_chunks': len(chunks),
                    'created_at': datetime.utcnow().isoformat(),
                    'chunker_config': {
                        'max_chunk_size': self.max_chunk_size,
                        'chunk_overlap': self.chunk_overlap
                    }
                },
                'chunks': [asdict(chunk) for chunk in chunks]
            }
            
            # Save to JSON with proper encoding
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(
                "Chunks saved to JSON",
                output_path=output_path,
                chunk_count=len(chunks),
                file_size=os.path.getsize(output_path)
            )
            
            log_function_result(self.logger, "save_chunks_to_json", result=output_path)
            return output_path
            
        except Exception as e:
            log_function_result(self.logger, "save_chunks_to_json", error=e)
            raise


# Convenience functions for external use
async def process_pdf_document(file_path: str) -> List[DocumentChunk]:
    """Process a single PDF document into chunks"""
    chunker = DocumentChunker()
    return await chunker.process_document(file_path)


async def process_all_documents(input_dir: str = "data/raw") -> List[DocumentChunk]:
    """Process all documents in a directory"""
    chunker = DocumentChunker()
    all_chunks = []
    
    logger = get_logger(__name__)
    log_function_call(logger, "process_all_documents", input_dir=input_dir)
    
    try:
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                try:
                    chunks = await chunker.process_document(file_path)
                    all_chunks.extend(chunks)
                    logger.info("Processed document", file=filename, chunks=len(chunks))
                except Exception as e:
                    logger.error("Failed to process document", file=filename, error=str(e))
                    continue
        
        logger.info("All documents processed", total_files=len(os.listdir(input_dir)), total_chunks=len(all_chunks))
        log_function_result(logger, "process_all_documents", result=f"{len(all_chunks)} total chunks")
        return all_chunks
        
    except Exception as e:
        log_function_result(logger, "process_all_documents", error=e)
        raise
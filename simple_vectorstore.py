#!/usr/bin/env python3
"""
Simple in-memory vector store for IUFP RAG demo
This provides a quick fallback when pgvector is not available
"""

import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import pickle
import os

@dataclass
class SimpleSearchResult:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    metadata: Optional[Dict] = None

class SimpleVectorStore:
    def __init__(self, storage_file: str = "data/simple_vectors.pkl"):
        self.storage_file = storage_file
        self.vectors = {}  # chunk_id -> {"embedding": [], "chunk": ChunkData}
        self.load_vectors()
    
    def load_vectors(self):
        """Load vectors from disk if available"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'rb') as f:
                    self.vectors = pickle.load(f)
                print(f"Loaded {len(self.vectors)} vectors from {self.storage_file}")
            except Exception as e:
                print(f"Error loading vectors: {e}")
                self.vectors = {}
    
    def save_vectors(self):
        """Save vectors to disk"""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            print(f"Saved {len(self.vectors)} vectors to {self.storage_file}")
        except Exception as e:
            print(f"Error saving vectors: {e}")
    
    def store_chunk_with_embedding(self, chunk, embedding: List[float]):
        """Store a chunk with its embedding"""
        self.vectors[chunk.chunk_id] = {
            "embedding": np.array(embedding),
            "chunk": {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
        }
        self.save_vectors()
    
    def store_chunks_batch(self, chunks_with_embeddings):
        """Store multiple chunks with embeddings"""
        for chunk, embedding in chunks_with_embeddings:
            self.vectors[chunk.chunk_id] = {
                "embedding": np.array(embedding),
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "text": chunk.text,
                    "metadata": chunk.metadata
                }
            }
        self.save_vectors()
        print(f"Stored {len(chunks_with_embeddings)} chunks in simple vector store")
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def similarity_search(self, query_embedding: List[float], limit: int = 10) -> List[SimpleSearchResult]:
        """Perform similarity search"""
        query_vec = np.array(query_embedding)
        
        similarities = []
        for chunk_id, data in self.vectors.items():
            similarity = self.cosine_similarity(query_vec, data["embedding"])
            similarities.append((chunk_id, similarity, data["chunk"]))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for chunk_id, score, chunk_data in similarities[:limit]:
            result = SimpleSearchResult(
                chunk_id=chunk_data["chunk_id"],
                document_id=chunk_data["document_id"],
                document_name=chunk_data["document_name"],
                text=chunk_data["text"],
                score=score,
                metadata=chunk_data.get("metadata")
            )
            results.append(result)
        
        return results
    
    def get_stats(self):
        """Get statistics about stored vectors"""
        return {
            "total_chunks": len(self.vectors),
            "unique_documents": len(set(v["chunk"]["document_id"] for v in self.vectors.values())),
            "storage_file": self.storage_file
        }

# Test the simple vector store
if __name__ == "__main__":
    store = SimpleVectorStore()
    print("Simple Vector Store loaded successfully!")
    print(f"Stats: {store.get_stats()}")
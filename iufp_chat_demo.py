#!/usr/bin/env python3
"""
IUFP RAG Chatbot Demo Server
Simple demonstration of the IUFP RAG system using the simple vector store
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, '.')

from simple_vectorstore import SimpleVectorStore
from src.embedder import EmbeddingService
from src.config import settings
from src.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize components
vector_store = SimpleVectorStore()
embedding_service = EmbeddingService()
openai_client = OpenAI(api_key=settings.openai_api_key)

app = FastAPI(
    title="IUFP RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for IUFP knowledge base",
    version="1.0.0"
)

# Add CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    max_results: int = 1
    include_sources: bool = True
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    processing_time: float
    session_id: str
    message_id: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vector_store_stats": stats
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = vector_store.get_stats()
    return {
        "system": "IUFP RAG Chatbot",
        "vector_store": stats,
        "models": {
            "embedding": settings.embedding_model,
            "chat": settings.chat_model
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    start_time = datetime.now()
    message_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info("Processing chat request", 
                   message_id=message_id, 
                   session_id=session_id,
                   message_length=len(request.message))
        
        # Step 1: Create embedding for user question
        query_embedding_result = await embedding_service.create_embedding(request.message)
        
        # Step 2: Retrieve relevant chunks
        search_results = vector_store.similarity_search(
            query_embedding_result.embedding, 
            limit=request.max_results
        )
        
        # Step 3: Prepare context from retrieved chunks
        context_chunks = []
        sources = []
        
        for result in search_results:
            context_chunks.append(f"Document: {result.document_name}\\nContent: {result.text}")
            if request.include_sources:
                sources.append(f"{result.document_name} (relevance: {result.score:.3f})")
        
        context = "\\n\\n".join(context_chunks)
        
        # Step 4: Generate response using OpenAI
        system_prompt = """You are an IUFP assistant. Provide concise, well-formatted answers using the context provided.

Formatting rules:
- Use simple bullet points (•) instead of numbers
- Keep sentences short and clear
- Separate main points with line breaks
- Avoid markdown formatting like ** or *
- Summarize information, don't repeat everything
- Focus on the most important details

When information is not available in the context:
- NEVER use phrases like "the document does not include", "not provided", "not available in the context"
- Instead, start with: "For this specific information, I recommend..."
- Guide users to visit www.iufp.com for comprehensive details
- Suggest booking a consultation through the IUFP website for personalized guidance
- Always remain helpful and professional"""

        user_prompt = f"""Context from IUFP knowledge base:
{context}

User question: {request.message}

Please provide a helpful and accurate response based on the context above."""

        chat_completion = openai_client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Chat response generated successfully",
                   message_id=message_id,
                   processing_time=processing_time,
                   response_length=len(response_text),
                   sources_count=len(sources))
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            processing_time=processing_time,
            session_id=session_id,
            message_id=message_id
        )
        
    except Exception as e:
        logger.error("Chat request failed", 
                    message_id=message_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/test-search")
async def test_search(query: str = "What is IUFP?", limit: int = 3):
    """Test search functionality"""
    try:
        query_embedding_result = await embedding_service.create_embedding(query)
        results = vector_store.similarity_search(query_embedding_result.embedding, limit=limit)
        
        return {
            "query": query,
            "results": [
                {
                    "document": result.document_name,
                    "score": result.score,
                    "text_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text
                }
                for result in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search test failed: {str(e)}")

if __name__ == "__main__":
    print("Starting IUFP RAG Chatbot Server...")
    print(f"Vector Store Stats: {vector_store.get_stats()}")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Test Search: http://localhost:8000/test-search")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "iufp_chat_demo:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
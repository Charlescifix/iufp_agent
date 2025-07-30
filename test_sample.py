#!/usr/bin/env python3
"""
Quick test script to verify the RAG system components
"""

import asyncio
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.config import settings
        print("✅ Config imported")
        
        from src.logger import get_logger
        print("✅ Logger imported")
        
        from src.chunker import DocumentChunker
        print("✅ Chunker imported")
        
        from src.embedder import EmbeddingService
        print("✅ Embedder imported")
        
        from src.vectorstore import PostgreSQLVectorStore
        print("✅ VectorStore imported")
        
        from src.retriever import HybridRetriever
        print("✅ Retriever imported")
        
        from src.ingestion import S3IngestionService
        print("✅ Ingestion imported")
        
        from src.chat_api import app
        print("✅ Chat API imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\nTesting basic functionality...")
    
    try:
        from src.config import settings
        from src.logger import get_logger, setup_logging
        
        # Test logging
        setup_logging()
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logging works")
        
        # Test config
        print(f"✅ Config loaded - Debug mode: {settings.debug}")
        print(f"✅ Embedding model: {settings.embedding_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("RAG System Component Test")
        print("=" * 40)
        
        imports_ok = await test_imports()
        basic_ok = await test_basic_functionality()
        
        if imports_ok and basic_ok:
            print("\n✅ All tests passed! System components are working.")
            print("\nNext steps:")
            print("1. Configure your .env file with API keys")
            print("2. Run: python quickstart.py setup")
            print("3. Add documents to data/raw/ folder")
            print("4. Run: python quickstart.py all")
        else:
            print("\n❌ Some tests failed. Check your environment and dependencies.")
    
    asyncio.run(main())
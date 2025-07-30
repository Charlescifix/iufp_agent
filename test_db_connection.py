#!/usr/bin/env python3
"""
Quick database connection test
Tests basic database connectivity and configuration
"""

import os
import sys
import asyncio
from datetime import datetime

# Add src to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration():
    """Test database configuration"""
    print("Testing database configuration...")
    
    try:
        from src.config import settings
        print("Config loaded successfully")
        
        # Check database URL
        if not settings.database_url:
            print("WARNING: DATABASE_URL not configured")
            print("Please set DATABASE_URL in your .env file")
            return False
        
        if not settings.database_url.startswith('postgresql://'):
            print("ERROR: Invalid database URL format")
            print("Expected: postgresql://user:password@host:port/database")
            return False
        
        print(f"Database URL configured: {settings.database_url[:20]}...")
        print(f"Embedding dimension: {settings.embedding_dimension}")
        print(f"Debug mode: {settings.debug}")
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def test_database_imports():
    """Test if database-related modules can be imported"""
    print("\nTesting database imports...")
    
    try:
        import psycopg2
        print("psycopg2 imported successfully")
        
        from sqlalchemy import create_engine
        print("SQLAlchemy imported successfully")
        
        try:
            import pgvector
            print("pgvector imported successfully")
        except ImportError:
            print("WARNING: pgvector not available - vector operations may fail")
        
        from src.vectorstore import PostgreSQLVectorStore, SearchResult, ChatMessage
        print("VectorStore classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

async def test_database_connection():
    """Test actual database connection"""
    print("\nTesting database connection...")
    
    try:
        from src.config import settings
        
        if not settings.database_url:
            print("SKIPPED: No database URL configured")
            return True
        
        from src.vectorstore import PostgreSQLVectorStore
        
        # Try to create vector store (this will test connection)
        vector_store = PostgreSQLVectorStore()
        print("Vector store created successfully")
        
        # Test basic query
        stats = await vector_store.get_document_stats()
        print(f"Database stats retrieved: {stats['total_chunks']} chunks, {stats['unique_documents']} documents")
        
        return True
        
    except Exception as e:
        print(f"Connection test failed: {e}")
        print("\nPossible issues:")
        print("1. Database server not running")
        print("2. Incorrect connection credentials")
        print("3. Database does not exist")
        print("4. pgvector extension not installed")
        print("5. Network connectivity issues")
        return False

def test_pgvector_extension():
    """Test if pgvector extension is available"""
    print("\nTesting pgvector extension...")
    
    try:
        from src.config import settings
        
        if not settings.database_url:
            print("SKIPPED: No database URL configured")
            return True
        
        import psycopg2
        
        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        
        # Check if pgvector extension exists
        cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        extension_exists = cur.fetchone()[0]
        
        if extension_exists:
            print("pgvector extension is installed")
        else:
            print("WARNING: pgvector extension is not installed")
            print("To install: CREATE EXTENSION vector; (requires superuser)")
        
        # Check vector operations
        if extension_exists:
            cur.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")
            distance = cur.fetchone()[0]
            print(f"Vector distance test successful: {distance}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"pgvector test failed: {e}")
        return False

async def main():
    """Run all connection tests"""
    print("Database Connection Test Suite")
    print("=" * 40)
    
    tests = [
        ("Configuration", test_configuration),
        ("Imports", test_database_imports),
        ("Connection", test_database_connection),
        ("pgvector Extension", test_pgvector_extension),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
            
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\nAll tests passed! Database is ready to use.")
    else:
        print("\nSome tests failed. Please check the configuration and database setup.")
        print("\nNext steps:")
        print("1. Ensure PostgreSQL is running")
        print("2. Verify database URL in .env file")
        print("3. Install pgvector extension: CREATE EXTENSION vector;")
        print("4. Check network connectivity to database")

if __name__ == "__main__":
    asyncio.run(main())
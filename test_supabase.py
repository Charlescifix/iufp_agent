#!/usr/bin/env python3
"""
Supabase Database Test Suite
Comprehensive tests for Supabase PostgreSQL with pgvector
"""

import os
import sys
import asyncio
import json
from datetime import datetime
import uuid
from typing import Dict, Any

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_environment_setup():
    """Test environment and configuration setup"""
    print("=== Environment Setup Test ===")
    
    try:
        from src.config import settings
        
        # Check for .env file
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            print("✓ .env file found")
        else:
            print("⚠ .env file not found - using environment variables")
        
        # Check database URL
        if not settings.database_url:
            print("✗ DATABASE_URL not configured")
            print("Please add DATABASE_URL to your .env file")
            return False
        
        # Parse Supabase URL
        if 'supabase.co' in settings.database_url:
            print("✓ Supabase database URL detected")
            # Extract project ID from URL
            try:
                url_parts = settings.database_url.split('@')[1].split('.')[0]
                project_id = url_parts.replace('db.', '')
                print(f"  Project ID: {project_id}")
            except:
                print("  Could not parse project ID")
        else:
            print("⚠ Not a Supabase URL - proceeding with generic PostgreSQL")
        
        print(f"✓ Embedding dimension: {settings.embedding_dimension}")
        print(f"✓ Debug mode: {settings.debug}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return False

def test_network_connectivity():
    """Test network connectivity to Supabase"""
    print("\n=== Network Connectivity Test ===")
    
    try:
        from src.config import settings
        import socket
        from urllib.parse import urlparse
        
        if not settings.database_url:
            print("⚠ No database URL to test")
            return True
        
        # Parse database URL
        parsed = urlparse(settings.database_url)
        host = parsed.hostname
        port = parsed.port or 5432
        
        print(f"Testing connection to {host}:{port}")
        
        # Test socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✓ Network connection successful")
            return True
        else:
            print(f"✗ Network connection failed (error code: {result})")
            print("Possible issues:")
            print("  - Internet connectivity problems")
            print("  - Firewall blocking connection")
            print("  - Supabase project paused/suspended")
            print("  - Incorrect host/port")
            return False
        
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False

async def test_database_connection():
    """Test actual database connection and authentication"""
    print("\n=== Database Connection Test ===")
    
    try:
        from src.config import settings
        import psycopg2
        
        if not settings.database_url:
            print("⚠ No database URL configured")
            return False
        
        print("Attempting database connection...")
        
        # Try basic connection
        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        
        # Test basic query
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        print(f"✓ Connected to: {version[:50]}...")
        
        # Check current user and database
        cur.execute("SELECT current_user, current_database()")
        user, database = cur.fetchone()
        print(f"✓ Connected as: {user}")
        print(f"✓ Database: {database}")
        
        # Check permissions
        cur.execute("""
            SELECT has_database_privilege(current_user, current_database(), 'CREATE'),
                   has_database_privilege(current_user, current_database(), 'CONNECT')
        """)
        can_create, can_connect = cur.fetchone()
        print(f"✓ Can connect: {can_connect}")
        print(f"✓ Can create objects: {can_create}")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("Possible issues:")
        print("  - Invalid credentials in DATABASE_URL")
        print("  - Database doesn't exist")
        print("  - User doesn't have access permissions")
        print("  - Connection string format error")
        return False

async def test_pgvector_extension():
    """Test pgvector extension availability"""
    print("\n=== pgvector Extension Test ===")
    
    try:
        from src.config import settings
        import psycopg2
        
        if not settings.database_url:
            print("⚠ No database URL configured")
            return False
        
        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        
        # Check if pgvector extension exists
        cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        extension_exists = cur.fetchone()[0]
        
        if extension_exists:
            print("✓ pgvector extension is installed")
            
            # Test vector operations
            cur.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")
            distance = cur.fetchone()[0]
            print(f"✓ Vector operations working (test distance: {distance})")
            
            # Check available vector functions
            cur.execute("""
                SELECT proname FROM pg_proc 
                WHERE proname LIKE '%vector%' 
                ORDER BY proname
            """)
            functions = [row[0] for row in cur.fetchall()]
            print(f"✓ Available vector functions: {len(functions)}")
            
        else:
            print("✗ pgvector extension not installed")
            print("In Supabase, you can enable it via:")
            print("  Dashboard > Settings > Database > Extensions")
            print("  Search for 'vector' and enable it")
        
        cur.close()
        conn.close()
        
        return extension_exists
        
    except Exception as e:
        print(f"✗ pgvector test failed: {e}")
        return False

async def test_table_creation():
    """Test table creation and schema setup"""
    print("\n=== Table Creation Test ===")
    
    try:
        from src.vectorstore import PostgreSQLVectorStore
        
        print("Creating vector store (this will create tables)...")
        vector_store = PostgreSQLVectorStore()
        print("✓ Vector store created successfully")
        
        # Check if tables were created
        stats = await vector_store.get_document_stats()
        print(f"✓ Tables accessible - {stats['total_chunks']} chunks in store")
        print(f"✓ Table size: {stats['table_size']}")
        print(f"✓ Chat table size: {stats['chat_table_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Table creation failed: {e}")
        return False

async def test_basic_operations():
    """Test basic database operations"""
    print("\n=== Basic Operations Test ===")
    
    try:
        from src.vectorstore import PostgreSQLVectorStore
        from src.chunker import DocumentChunk
        from src.config import settings
        
        vector_store = PostgreSQLVectorStore()
        
        # Create test chunk
        test_chunk = DocumentChunk(
            chunk_id=f"test_{uuid.uuid4()}",
            document_id=f"doc_{uuid.uuid4()}",
            document_name="test_document.pdf",
            page_number=1,
            chunk_index=0,
            text="This is a test chunk for Supabase database testing.",
            char_count=52,
            word_count=10,
            source_hash="test_hash",
            created_at=datetime.now().isoformat(),
            section_title="Test Section",
            metadata={"test": True, "environment": "supabase"}
        )
        
        # Create test embedding
        test_embedding = [0.1] * settings.embedding_dimension
        
        print("Testing chunk storage...")
        await vector_store.store_chunk_with_embedding(test_chunk, test_embedding)
        print("✓ Chunk stored successfully")
        
        print("Testing similarity search...")
        results = await vector_store.similarity_search(test_embedding, limit=5)
        print(f"✓ Search returned {len(results)} results")
        
        # Find our test chunk
        test_result = next((r for r in results if r.chunk_id == test_chunk.chunk_id), None)
        if test_result:
            print(f"✓ Test chunk found with score: {test_result.score:.3f}")
        
        print("Testing document deletion...")
        deleted_count = await vector_store.delete_document(test_chunk.document_id)
        print(f"✓ Deleted {deleted_count} chunks")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        return False

async def test_chat_functionality():
    """Test chat message storage and retrieval"""
    print("\n=== Chat Functionality Test ===")
    
    try:
        from src.vectorstore import PostgreSQLVectorStore, ChatMessage
        
        vector_store = PostgreSQLVectorStore()
        
        # Create test chat message
        test_message = ChatMessage(
            message_id=f"msg_{uuid.uuid4()}",
            session_id=f"session_{uuid.uuid4()}",
            user_message="How do I apply for a UK student visa?",
            bot_response="To apply for a UK student visa, you need to...",
            sources=["IUFP-Tier4StudentVisaChecklist.pdf"],
            created_at=datetime.now().isoformat(),
            processing_time=2.5,
            token_count=150
        )
        
        print("Testing chat message storage...")
        await vector_store.store_chat_message(test_message, user_ip="127.0.0.1")
        print("✓ Chat message stored")
        
        print("Testing chat history retrieval...")
        history = await vector_store.get_chat_history(test_message.session_id, limit=10)
        print(f"✓ Retrieved {len(history)} messages")
        
        # Verify our message
        found_message = next((m for m in history if m.message_id == test_message.message_id), None)
        if found_message:
            print("✓ Test message found in history")
        
        return True
        
    except Exception as e:
        print(f"✗ Chat functionality failed: {e}")
        return False

def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate comprehensive test report"""
    print("\n" + "=" * 60)
    print("SUPABASE DATABASE TEST REPORT")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Tests Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 40)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        emoji = "✓" if result else "✗"
        print(f"{emoji} {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Supabase database is ready for use.")
        print("\nNext steps:")
        print("1. Run: python quickstart.py setup")
        print("2. Add documents to data/raw/ folder")
        print("3. Run: python quickstart.py all")
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("Please review the failed tests above.")
        print("\nCommon fixes:")
        print("1. Check your DATABASE_URL in .env")
        print("2. Ensure internet connectivity")
        print("3. Enable pgvector extension in Supabase dashboard")
        print("4. Verify your Supabase project is active")

async def main():
    """Run comprehensive Supabase database tests"""
    print("SUPABASE DATABASE TEST SUITE")
    print("=" * 60)
    
    tests = {
        "Environment Setup": test_environment_setup,
        "Network Connectivity": test_network_connectivity,
        "Database Connection": test_database_connection,
        "pgvector Extension": test_pgvector_extension,
        "Table Creation": test_table_creation,
        "Basic Operations": test_basic_operations,
        "Chat Functionality": test_chat_functionality
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results[test_name] = result
            
        except Exception as e:
            print(f"\n✗ {test_name} encountered an error: {e}")
            results[test_name] = False
    
    generate_test_report(results)

if __name__ == "__main__":
    asyncio.run(main())
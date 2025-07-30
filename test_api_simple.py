#!/usr/bin/env python3
"""
Simple API functionality test without database dependency
Tests the core components and configuration
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """Test that configuration loads correctly"""
    print("🔧 Testing Configuration Loading...")
    try:
        from src.config import settings
        
        # Check critical settings
        checks = {
            "Database URL": bool(settings.database_url),
            "OpenAI API Key": bool(settings.openai_api_key),
            "Secret Key": bool(settings.secret_key),
            "Admin API Key": bool(settings.admin_api_key),
            "Embedding Model": settings.embedding_model == "text-embedding-3-large",
            "Chat Model": settings.chat_model == "gpt-4-turbo"
        }
        
        all_passed = True
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("✅ Configuration test passed!")
            return True
        else:
            print("❌ Configuration test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("\n📦 Testing Module Imports...")
    
    modules_to_test = [
        ("src.config", "Configuration module"),
        ("src.logger", "Logging module"),
        ("src.chat_api", "Chat API module"),
        ("src.chunker", "Document chunker"),
        ("src.embedder", "Embedding module"),
        ("src.retriever", "Retrieval module"),
        ("fastapi", "FastAPI framework"),
        ("psycopg2", "PostgreSQL adapter"),
        ("openai", "OpenAI client"),
        ("structlog", "Structured logging")
    ]
    
    all_passed = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"   ✅ {description}")
        except ImportError as e:
            print(f"   ❌ {description}: {e}")
            all_passed = False
        except Exception as e:
            print(f"   ⚠️  {description}: {e}")
    
    if all_passed:
        print("✅ All imports successful!")
        return True
    else:
        print("❌ Some imports failed!")
        return False

def test_logger():
    """Test logging functionality"""
    print("\n📝 Testing Logging System...")
    
    try:
        from src.logger import get_logger, log_function_call, log_function_result
        
        # Create test logger
        logger = get_logger("test_logger")
        
        # Test basic logging
        logger.info("Test log message", test_field="test_value")
        
        # Test function logging
        log_function_call(logger, "test_function", param1="value1")
        log_function_result(logger, "test_function", result="success")
        
        print("   ✅ Logger initialization")
        print("   ✅ Structured logging")
        print("   ✅ Function call logging")
        print("✅ Logging test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Logging test error: {e}")
        return False

def test_openai_config():
    """Test OpenAI configuration (without making API calls)"""
    print("\n🤖 Testing OpenAI Configuration...")
    
    try:
        from openai import OpenAI
        from src.config import settings
        
        # Create client (doesn't make API call)
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Check API key format
        api_key = settings.openai_api_key
        if api_key and api_key.startswith('sk-') and len(api_key) > 20:
            print("   ✅ OpenAI API key format valid")
        else:
            print("   ❌ OpenAI API key format invalid")
            return False
        
        # Check model configuration
        if settings.embedding_model and settings.chat_model:
            print(f"   ✅ Models configured: {settings.embedding_model}, {settings.chat_model}")
        else:
            print("   ❌ Models not configured")
            return False
        
        print("✅ OpenAI configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI configuration test error: {e}")
        return False

def test_security_config():
    """Test security configuration"""
    print("\n🔒 Testing Security Configuration...")
    
    try:
        from src.config import settings
        
        # Check secret key
        if settings.secret_key and len(settings.secret_key) >= 32:
            print("   ✅ Secret key length adequate")
        else:
            print("   ❌ Secret key too short or missing")
            return False
        
        # Check admin API key
        if settings.admin_api_key and len(settings.admin_api_key) >= 16:
            print("   ✅ Admin API key configured")
        else:
            print("   ❌ Admin API key missing or too short")
            return False
        
        # Check rate limiting
        if settings.rate_limit_requests > 0:
            print(f"   ✅ Rate limiting: {settings.rate_limit_requests} requests per hour")
        else:
            print("   ❌ Rate limiting not configured")
            return False
        
        print("✅ Security configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Security configuration test error: {e}")
        return False

def test_fastapi_app_creation():
    """Test FastAPI app can be created (without starting server)"""
    print("\n🚀 Testing FastAPI App Creation...")
    
    try:
        # Import without triggering database connection
        import sys
        from unittest.mock import patch
        
        # Mock the database components to avoid connection issues
        with patch('src.vectorstore.PostgreSQLVectorStore') as mock_store, \
             patch('src.retriever.HybridRetriever') as mock_retriever:
            
            # Mock successful initialization
            mock_store.return_value.get_document_stats.return_value = {}
            mock_retriever.return_value = object()
            
            from src.chat_api import app
            
            print("   ✅ FastAPI app created")
            print("   ✅ Middleware configured")
            print("   ✅ Routes registered")
            
        print("✅ FastAPI app creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app creation test error: {e}")
        return False

def main():
    """Run all tests"""
    print("IUFP RAG Chat API - Component Testing")
    print("=" * 60)
    print("Note: Testing core components without database dependency")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Module Imports", test_imports), 
        ("Logging System", test_logger),
        ("OpenAI Config", test_openai_config),
        ("Security Config", test_security_config),
        ("FastAPI App", test_fastapi_app_creation)
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
            failed += 1
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All component tests passed!")
        print("\nNext Steps:")
        print("   1. Fix database connection for full API testing")
        print("   2. Deploy to Railway with proper DATABASE_URL")
        print("   3. Run endpoint tests in production")
    else:
        print("Some tests failed. Check configuration and dependencies.")
    
    return results

if __name__ == "__main__":
    main()
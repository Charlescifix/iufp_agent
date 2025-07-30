#!/usr/bin/env python3
"""
Simple component test for IUFP RAG Chat API
Tests core functionality without database dependency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """Test configuration loading"""
    print("Testing Configuration Loading...")
    try:
        from src.config import settings
        
        checks = {
            "Database URL": bool(settings.database_url),
            "OpenAI API Key": bool(settings.openai_api_key),
            "Secret Key": bool(settings.secret_key),
            "Admin API Key": bool(settings.admin_api_key),
        }
        
        all_passed = True
        for check, result in checks.items():
            status = "PASS" if result else "FAIL"
            print(f"   {check}: {status}")
            if not result:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"Configuration test error: {e}")
        return False

def test_imports():
    """Test module imports"""
    print("Testing Module Imports...")
    
    modules = [
        "src.config",
        "src.logger", 
        "src.chat_api",
        "fastapi",
        "psycopg2",
        "openai"
    ]
    
    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"   {module}: PASS")
        except ImportError as e:
            print(f"   {module}: FAIL - {e}")
            all_passed = False
    
    return all_passed

def test_logger():
    """Test logging system"""
    print("Testing Logging System...")
    
    try:
        from src.logger import get_logger
        logger = get_logger("test")
        logger.info("Test message")
        print("   Logging: PASS")
        return True
    except Exception as e:
        print(f"   Logging: FAIL - {e}")
        return False

def main():
    """Run all tests"""
    print("IUFP RAG Chat API - Component Testing")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("Logging", test_logger)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"{test_name}: OVERALL PASS")
            else:
                failed += 1
                print(f"{test_name}: OVERALL FAIL")
        except Exception as e:
            failed += 1
            print(f"{test_name}: CRASH - {e}")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All component tests passed!")
    else:
        print("Some tests failed.")

if __name__ == "__main__":
    main()
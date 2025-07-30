#!/usr/bin/env python3
"""
Test script for IUFP RAG Chat API endpoints
Tests health, chat functionality, and monitoring endpoints
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"
ADMIN_API_KEY = "iufp-admin-api-key-2024-secure-access"  # From .env file

def test_health_endpoint() -> Dict[str, Any]:
    """Test the health check endpoint"""
    print("\n🏥 Testing Health Endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check Passed")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Components: {json.dumps(data.get('components', {}), indent=2)}")
            return {"success": True, "data": data}
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"success": False, "error": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health Check Error: {str(e)}")
        return {"success": False, "error": str(e)}

def test_chat_endpoint() -> Dict[str, Any]:
    """Test the chat endpoint with a sample question"""
    print("\n💬 Testing Chat Endpoint...")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": ADMIN_API_KEY
    }
    
    payload = {
        "message": "What is IUFP and what services do you provide?",
        "session_id": f"test_session_{int(time.time())}",
        "include_sources": True,
        "max_results": 5
    }
    
    try:
        print(f"Sending request: {payload['message']}")
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat Request Successful")
            print(f"   Message ID: {data.get('message_id')}")
            print(f"   Session ID: {data.get('session_id')}")
            print(f"   Processing Time: {data.get('processing_time', 0):.2f}s")
            print(f"   Response Length: {len(data.get('response', ''))}")
            print(f"   Sources Count: {len(data.get('sources', []))}")
            print(f"   Response Preview: {data.get('response', '')[:200]}...")
            return {"success": True, "data": data}
        else:
            print(f"❌ Chat Request Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"success": False, "error": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat Request Error: {str(e)}")
        return {"success": False, "error": str(e)}

def test_stats_endpoint() -> Dict[str, Any]:
    """Test the statistics endpoint (admin only)"""
    print("\n📊 Testing Stats Endpoint...")
    
    headers = {
        "X-API-Key": ADMIN_API_KEY
    }
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/stats", 
            headers=headers,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats Request Successful")
            print(f"   Total Messages: {data.get('total_messages', 0)}")
            print(f"   Active Sessions: {data.get('active_sessions', 0)}")
            print(f"   Avg Response Time: {data.get('avg_response_time', 0):.2f}s")
            print(f"   System Stats: {json.dumps(data.get('system_stats', {}), indent=2)}")
            return {"success": True, "data": data}
        else:
            print(f"❌ Stats Request Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"success": False, "error": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats Request Error: {str(e)}")
        return {"success": False, "error": str(e)}

def test_unauthorized_access() -> Dict[str, Any]:
    """Test unauthorized access (should fail)"""
    print("\n🔒 Testing Unauthorized Access...")
    
    payload = {
        "message": "This should fail without API key",
        "session_id": "unauthorized_test"
    }
    
    try:
        # No API key provided
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print(f"✅ Unauthorized Access Properly Blocked")
            return {"success": True, "message": "Security working correctly"}
        else:
            print(f"❌ Security Issue: Request should have been blocked")
            print(f"   Response: {response.text}")
            return {"success": False, "error": "Security bypass detected"}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Unauthorized Test Error: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    """Run all API endpoint tests"""
    print("🧪 IUFP RAG Chat API Endpoint Testing")
    print("=" * 50)
    
    # Test results
    results = {
        "health": test_health_endpoint(),
        "chat": test_chat_endpoint(),
        "stats": test_stats_endpoint(),
        "security": test_unauthorized_access()
    }
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{test_name.upper():12} {status}")
        if result["success"]:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return results

if __name__ == "__main__":
    main()
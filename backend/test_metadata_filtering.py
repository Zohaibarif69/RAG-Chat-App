#!/usr/bin/env python3
"""
Test script for metadata filtering functionality
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_metadata_filters():
    """Test getting available metadata filters"""
    print("\n" + "="*60)
    print("TEST 1: Get available metadata filters")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/metadata-filters")
        data = response.json()
        
        if data.get("success"):
            print("✓ Available filters:")
            print(json.dumps(data.get("filters"), indent=2))
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Connection error: {e}")

def test_query_with_filters():
    """Test querying with metadata filters"""
    print("\n" + "="*60)
    print("TEST 2: Query with metadata filters")
    print("="*60)
    
    query = {
        "question": "What is the document about?",
        "session_id": "test-session-123",
        "top_k": 5,
        "filters": {
            # Uncomment filters to test with actual data
            # "category": "technical",
            # "year": "2026",
            # "filename": "your-file.pdf"
        }
    }
    
    print(f"Query: {json.dumps(query, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=query)
        data = response.json()
        
        if data.get("success"):
            print(f"\n✓ Answer:\n{data.get('answer')}")
            print(f"\nSources with metadata:")
            for source in data.get("sources", []):
                print(f"  - {source['filename']} (Page {source['page']}, {source['category']}, {source['year']}) - Relevance: {source['relevance']:.2%}")
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_query_by_category():
    """Test filtering by category"""
    print("\n" + "="*60)
    print("TEST 3: Query filtered by category")
    print("="*60)
    
    query = {
        "question": "What are the key points?",
        "session_id": "test-session-456",
        "filters": {
            "category": "technical"
        }
    }
    
    print(f"Query (category='technical'): {json.dumps(query, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=query)
        data = response.json()
        
        if data.get("success"):
            print(f"✓ Found {len(data.get('sources', []))} results")
            for source in data.get("sources", []):
                print(f"  - {source['filename']} (Category: {source['category']})")
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_query_by_file():
    """Test filtering by specific file"""
    print("\n" + "="*60)
    print("TEST 4: Query filtered by filename")
    print("="*60)
    
    query = {
        "question": "Explain this concept",
        "session_id": "test-session-789",
        "filters": {
            "filename": "sample.pdf"  # Replace with actual filename
        }
    }
    
    print(f"Query (filename='sample.pdf'): {json.dumps(query, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=query)
        data = response.json()
        
        if data.get("success"):
            print(f"✓ Found {len(data.get('sources', []))} results from specified file")
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_query_by_page():
    """Test filtering by page number"""
    print("\n" + "="*60)
    print("TEST 5: Query filtered by page number")
    print("="*60)
    
    query = {
        "question": "What is on page 1?",
        "session_id": "test-session-101",
        "filters": {
            "page": 1
        }
    }
    
    print(f"Query (page=1): {json.dumps(query, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=query)
        data = response.json()
        
        if data.get("success"):
            print(f"✓ Found {len(data.get('sources', []))} results from page 1")
            for source in data.get("sources", []):
                print(f"  - Page {source['page']} (Relevance: {source['relevance']:.2%})")
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_combined_filters():
    """Test combining multiple filters"""
    print("\n" + "="*60)
    print("TEST 6: Query with combined filters")
    print("="*60)
    
    query = {
        "question": "What is important?",
        "session_id": "test-session-202",
        "filters": {
            "category": "technical",
            "year": "2026",
            # "filename": "document.pdf"
        }
    }
    
    print(f"Query with multiple filters: {json.dumps(query, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=query)
        data = response.json()
        
        if data.get("success"):
            print(f"✓ Found {len(data.get('sources', []))} results matching all filters")
            for source in data.get("sources", []):
                print(f"  - {source['filename']} ({source['category']}, {source['year']}, Page {source['page']})")
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("METADATA FILTERING TEST SUITE")
    print("="*60)
    
    # Run all tests
    test_metadata_filters()
    
    # Optional: Run other tests only if filters are available
    # test_query_with_filters()
    # test_query_by_category()
    # test_query_by_file()
    # test_query_by_page()
    # test_combined_filters()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
    print("\nNote: Run 'test_metadata_filters()' first to see available options")
    print("Then edit the other test functions with real values and run them")

#!/usr/bin/env python3
"""
Simple integration test for metadata filtering using HTTP API calls
This avoids heavy imports and tests the actual deployed system
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def create_simple_pdf(text: str) -> bytes:
    """Create a minimal valid PDF"""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length """ + str(len(text) + 50).encode() + b""" >>
stream
BT
/F1 12 Tf
100 700 Td
(""" + text.encode() + b""") Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000229 00000 n 
0000000308 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
""" + str(len(b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length """)).encode() + b"""
%%EOF
"""

def test_health():
    """Test 1: Check if backend is running"""
    print_header("TEST 1: Backend Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Backend is running: {response.json()}")
            return True
        else:
            print_error(f"Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to backend at {BASE_URL}")
        print_info("Make sure backend is running: cd backend && uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_clear_db():
    """Test 2: Clear database"""
    print_header("TEST 2: Clear Database")
    
    try:
        response = requests.post(f"{BASE_URL}/clear-db", timeout=TIMEOUT)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print_success("Database cleared")
                return True
            else:
                print_error(f"Clear DB failed: {result.get('error')}")
                return False
        else:
            print_error(f"Clear DB returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Clear DB failed: {e}")
        return False

def test_upload_with_metadata():
    """Test 3: Upload documents with metadata"""
    print_header("TEST 3: Upload Documents with Metadata")
    
    documents = [
        ("Q1_2024_Finance.pdf", "Q1 2024 Financial Performance Revenue 10M Profit 2M", "finance", "2024"),
        ("Q2_2024_Finance.pdf", "Q2 2024 Financial Report Revenue 12M Profit 2.5M", "finance", "2024"),
        ("HR_2023_Annual.pdf", "2023 Annual HR Report Headcount 500 Salary Budget 50M", "hr", "2023"),
        ("Product_2024_Roadmap.pdf", "2024 Product Roadmap Feature A Feature B Feature C", "product", "2024"),
    ]
    
    results = []
    for filename, content, category, year in documents:
        try:
            print(f"\nUploading {filename} ({category}, {year})...")
            pdf_bytes = create_simple_pdf(content)
            
            files = {"file": (filename, pdf_bytes, "application/pdf")}
            params = {"category": category, "year": year}
            
            response = requests.post(
                f"{BASE_URL}/upload",
                files=files,
                params=params,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print_success(f"{filename}: {result['message']}")
                    results.append(True)
                else:
                    print_error(f"{filename}: {result.get('error')}")
                    results.append(False)
            else:
                print_error(f"{filename}: HTTP {response.status_code}")
                results.append(False)
        except Exception as e:
            print_error(f"{filename}: {e}")
            results.append(False)
        
        time.sleep(1)  # Rate limit
    
    return all(results) if results else False

def test_metadata_filters_endpoint():
    """Test 4: Get available metadata filters"""
    print_header("TEST 4: Get Available Metadata Filters")
    
    try:
        response = requests.get(f"{BASE_URL}/metadata-filters", timeout=TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                filters = result.get("filters", {})
                print_success(f"Retrieved metadata filters")
                print(f"\n{BOLD}Files:{RESET}")
                for f in filters.get("files", []):
                    print(f"  • {f}")
                print(f"\n{BOLD}Categories:{RESET}")
                for c in filters.get("categories", []):
                    print(f"  • {c}")
                print(f"\n{BOLD}Years:{RESET}")
                for y in filters.get("years", []):
                    print(f"  • {y}")
                return True
            else:
                print_error(f"Failed: {result.get('error')}")
                return False
        else:
            print_error(f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed: {e}")
        return False

def test_query_no_filter():
    """Test 5: Query without filters"""
    print_header("TEST 5: Query Without Filters")
    
    try:
        payload = {
            "question": "What financial reports do we have?",
            "session_id": "test-session-1",
            "top_k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print_success(f"Query returned results")
                sources = result.get("sources", [])
                print(f"\nFound {len(sources)} sources:")
                for i, src in enumerate(sources, 1):
                    print(f"  {i}. {src.get('filename')} (p.{src.get('page')}) - {src.get('category')} {src.get('year')}")
                return len(sources) > 0
            else:
                print_error(f"Query failed: {result.get('error')}")
                return False
        else:
            print_error(f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Query failed: {e}")
        return False

def test_filter_year_2024():
    """Test 6: Filter by year 2024"""
    print_header("TEST 6: Query with Year Filter (2024)")
    
    try:
        payload = {
            "question": "What happened in 2024?",
            "session_id": "test-session-2",
            "top_k": 5,
            "filters": {"year": "2024"}
        }
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                sources = result.get("sources", [])
                print_success(f"Found {len(sources)} sources for year=2024")
                
                all_2024 = True
                for i, src in enumerate(sources, 1):
                    year = src.get('year')
                    is_2024 = year == "2024"
                    status = "✓" if is_2024 else "✗"
                    print(f"  {i}. {src.get('filename')} - Year: {year} {status}")
                    if not is_2024:
                        all_2024 = False
                
                if all_2024:
                    print_success("All results are from 2024!")
                else:
                    print_warning("Some results are not from 2024")
                
                return len(sources) > 0 and all_2024
            else:
                print_error(f"Query failed: {result.get('error')}")
                return False
        else:
            print_error(f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Query failed: {e}")
        return False

def test_filter_category_finance():
    """Test 7: Filter by category finance"""
    print_header("TEST 7: Query with Category Filter (finance)")
    
    try:
        payload = {
            "question": "What are the financial metrics?",
            "session_id": "test-session-3",
            "top_k": 5,
            "filters": {"category": "finance"}
        }
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                sources = result.get("sources", [])
                print_success(f"Found {len(sources)} sources for category=finance")
                
                all_finance = True
                for i, src in enumerate(sources, 1):
                    category = src.get('category')
                    is_finance = category == "finance"
                    status = "✓" if is_finance else "✗"
                    print(f"  {i}. {src.get('filename')} - Category: {category} {status}")
                    if not is_finance:
                        all_finance = False
                
                if all_finance:
                    print_success("All results are from finance category!")
                else:
                    print_warning("Some results are not from finance category")
                
                return len(sources) > 0 and all_finance
            else:
                print_error(f"Query failed: {result.get('error')}")
                return False
        else:
            print_error(f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Query failed: {e}")
        return False

def test_filter_combined():
    """Test 8: Filter by year AND category"""
    print_header("TEST 8: Query with Combined Filters (2024 + finance)")
    
    try:
        payload = {
            "question": "What about 2024 finance?",
            "session_id": "test-session-4",
            "top_k": 5,
            "filters": {"year": "2024", "category": "finance"}
        }
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                sources = result.get("sources", [])
                print_success(f"Found {len(sources)} sources for combined filter")
                
                all_match = True
                for i, src in enumerate(sources, 1):
                    year = src.get('year')
                    category = src.get('category')
                    year_ok = year == "2024"
                    cat_ok = category == "finance"
                    status = "✓" if (year_ok and cat_ok) else "✗"
                    print(f"  {i}. {src.get('filename')} - Year: {year} {'✓' if year_ok else '✗'}, Category: {category} {'✓' if cat_ok else '✗'} {status}")
                    if not (year_ok and cat_ok):
                        all_match = False
                
                if all_match:
                    print_success("All results match both filters!")
                else:
                    print_warning("Some results don't match all filters")
                
                return len(sources) > 0 and all_match
            else:
                print_error(f"Query failed: {result.get('error')}")
                return False
        else:
            print_error(f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Query failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print_header("METADATA FILTERING SYSTEM - INTEGRATION TEST")
    
    tests = [
        ("Backend Health", test_health),
        ("Clear Database", test_clear_db),
        ("Upload Documents", test_upload_with_metadata),
        ("Get Metadata Filters", test_metadata_filters_endpoint),
        ("Query (No Filter)", test_query_no_filter),
        ("Filter by Year 2024", test_filter_year_2024),
        ("Filter by Category finance", test_filter_category_finance),
        ("Combined Filters", test_filter_combined),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except KeyboardInterrupt:
            print_error("Test interrupted")
            results[test_name] = False
            break
        except Exception as e:
            print_error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = f"{GREEN}PASS{RESET}" if passed_flag else f"{RED}FAIL{RESET}"
        print(f"{status} - {test_name}")
    
    print(f"\n{BOLD}Total: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ ALL TESTS PASSED - METADATA FILTERING WORKS!{RESET}")
    else:
        print(f"\n{YELLOW}{BOLD}⚠ {total - passed} tests failed{RESET}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

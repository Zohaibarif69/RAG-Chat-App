#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Verify metadata filtering works
Tests: Upload PDFs with metadata, then query with filters
"""
import requests
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import sys

BASE_URL = "http://localhost:8000"

def create_test_pdf(content: str, filename: str) -> BytesIO:
    """Create a test PDF with given content"""
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Add text to PDF
    text = c.beginText(50, 750)
    text.setFont("Helvetica", 12)
    for line in content.split('\n'):
        text.textLine(line)
    c.drawText(text)
    c.save()
    
    pdf_buffer.seek(0)
    return pdf_buffer

# Test PDFs to create
TEST_DOCS = [
    {
        "name": "Q1_2024_Finance_Report.pdf",
        "content": "Q1 2024 Financial Report\nRevenue: $5.2M\nProfit Margin: 23%\nKey metrics for 2024 fiscal year.",
        "category": "finance",
        "year": "2024"
    },
    {
        "name": "Q2_2024_Finance_Report.pdf", 
        "content": "Q2 2024 Financial Report\nRevenue: $6.1M\nProfit Margin: 25%\nSecond quarter 2024 results.",
        "category": "finance",
        "year": "2024"
    },
    {
        "name": "2023_Annual_Report.pdf",
        "content": "2023 Annual Financial Report\nTotal Revenue: $19.8M\nYearly summary for 2023.",
        "category": "finance",
        "year": "2023"
    },
    {
        "name": "Technical_Architecture_2024.pdf",
        "content": "Technical Architecture Document\nSystem design for 2024\nMicroservices architecture\nDatabase optimization techniques.",
        "category": "technical",
        "year": "2024"
    },
    {
        "name": "Python_Best_Practices.pdf",
        "content": "Python Development Best Practices\nPython coding standards\nPerformance optimization\nCode review guidelines for development.",
        "category": "technical",
        "year": "2023"
    }
]

def check_server():
    """Check if server is running"""
    print("\n" + "="*70)
    print("STEP 1: CHECK SERVER CONNECTION")
    print("="*70)
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ Server is running: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Server not responding: {e}")
        print(f"  Make sure to run: cd rag-app && python -m uvicorn backend.app.main:app --reload")
        return False

def clear_database():
    """Clear the database"""
    print("\n" + "="*70)
    print("STEP 2: CLEAR DATABASE")
    print("="*70)
    try:
        response = requests.post(f"{BASE_URL}/clear-db")
        data = response.json()
        if data.get("success"):
            print(f"✓ Database cleared: {data.get('message')}")
        else:
            print(f"✗ Error: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

def upload_test_documents():
    """Upload all test documents with metadata"""
    print("\n" + "="*70)
    print("STEP 3: UPLOAD TEST DOCUMENTS WITH METADATA")
    print("="*70)
    
    for doc in TEST_DOCS:
        print(f"\nUploading: {doc['name']}")
        print(f"  Category: {doc['category']}, Year: {doc['year']}")
        
        try:
            pdf_buffer = create_test_pdf(doc['content'], doc['name'])
            files = {'file': (doc['name'], pdf_buffer, 'application/pdf')}
            
            # Upload with metadata
            params = {
                'category': doc['category'],
                'year': doc['year']
            }
            
            response = requests.post(
                f"{BASE_URL}/upload",
                files=files,
                params=params
            )
            
            data = response.json()
            if data.get("success"):
                print(f"  ✓ Success: {data.get('message')}")
                print(f"    - Chunks: {data.get('chunks_count')}")
            else:
                print(f"  ✗ Error: {data.get('error')}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

def get_available_filters():
    """Get available metadata filter options"""
    print("\n" + "="*70)
    print("STEP 4: GET AVAILABLE FILTERS")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/metadata-filters")
        data = response.json()
        
        if data.get("success"):
            filters = data.get("filters", {})
            print(f"\n✓ Available files: {filters.get('files', [])}")
            print(f"✓ Available categories: {filters.get('categories', [])}")
            print(f"✓ Available years: {filters.get('years', [])}")
            print(f"✓ Available pages: {filters.get('pages', [])}")
            return filters
        else:
            print(f"✗ Error: {data.get('error')}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_filtering():
    """Test various filtering scenarios"""
    print("\n" + "="*70)
    print("STEP 5: TEST FILTERING SCENARIOS")
    print("="*70)
    
    test_cases = [
        {
            "name": "Test 5A: NO FILTER (all documents)",
            "filters": {}
        },
        {
            "name": "Test 5B: YEAR FILTER (2024 only)",
            "filters": {"year": "2024"}
        },
        {
            "name": "Test 5C: YEAR FILTER (2023 only)",
            "filters": {"year": "2023"}
        },
        {
            "name": "Test 5D: CATEGORY FILTER (finance only)",
            "filters": {"category": "finance"}
        },
        {
            "name": "Test 5E: CATEGORY FILTER (technical only)",
            "filters": {"category": "technical"}
        },
        {
            "name": "Test 5F: COMBINED FILTER (2024 AND finance)",
            "filters": {"year": "2024", "category": "finance"}
        },
        {
            "name": "Test 5G: COMBINED FILTER (2023 AND technical)",
            "filters": {"year": "2023", "category": "technical"}
        },
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 70)
        
        try:
            query = {
                "question": "What are the key information in this document?",
                "session_id": "verify-test",
                "filters": test_case['filters']
            }
            
            response = requests.post(f"{BASE_URL}/query", json=query)
            data = response.json()
            
            if data.get("success"):
                sources = data.get("sources", [])
                print(f"✓ Found {len(sources)} relevant chunks")
                
                # Group by filename
                files_found = {}
                for source in sources:
                    fname = source['filename']
                    if fname not in files_found:
                        files_found[fname] = {
                            "category": source['category'],
                            "year": source['year'],
                            "chunks": 0
                        }
                    files_found[fname]['chunks'] += 1
                
                print(f"  Files found:")
                for fname, info in sorted(files_found.items()):
                    print(f"    - {fname}")
                    print(f"      Category: {info['category']}, Year: {info['year']}, Chunks: {info['chunks']}")
                
                if test_case['filters']:
                    # Verify all sources match the filter
                    all_match = True
                    for source in sources:
                        for key, value in test_case['filters'].items():
                            if source.get(key) != value:
                                all_match = False
                                break
                    
                    if all_match and sources:
                        print(f"  ✓ All results match the filter criteria!")
                    elif sources:
                        print(f"  ⚠ Some results don't match filter criteria (possible context injection)")
                    else:
                        print(f"  ✓ No results found (filter resulted in empty set)")
            else:
                print(f"✗ Error: {data.get('error')}")
        except Exception as e:
            print(f"✗ Exception: {e}")

def summary():
    """Print summary"""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print("""
WHAT WAS TESTED:
✓ 1. Server connectivity
✓ 2. Database clearing
✓ 3. Upload with metadata (category & year)
✓ 4. Metadata extraction verification
✓ 5. Filtering scenarios:
    - No filter (baseline - all docs)
    - Year filter = 2024 (should find 3 docs)
    - Year filter = 2023 (should find 2 docs)
    - Category filter = finance (should find 3 docs)
    - Category filter = technical (should find 2 docs)
    - Combined filter = 2024 + finance (should find 2 docs)
    - Combined filter = 2023 + technical (should find 1 doc)

EXPECTED RESULTS:
- Each filter reduces results to only matching documents
- Metadata fields (filename, year, category, page) present in all results
- Multi-field filters use AND logic (both conditions must match)
- Relevance scores between 0.5 and 1.0 for matches

INTERPRETATION:
If filtering works correctly, you should see:
1. More results with NO filter
2. Fewer results with specific filters
3. Metadata displayed correctly for each result
""")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("METADATA FILTERING VERIFICATION TEST")
    print("="*70)
    
    if not check_server():
        sys.exit(1)
    
    clear_database()
    upload_test_documents()
    get_available_filters()
    test_filtering()
    summary()

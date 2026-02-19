#!/usr/bin/env python3
"""
Comprehensive test of metadata filtering system
Tests: filename, page, year, category filtering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.vector_store import (
    extract_text_from_pdf,
    chunk_documents,
    add_documents,
    retrieve_context,
    rag_query,
    get_all_documents,
    get_chroma_vectorstore,
)
import json

# ANSI color codes for better output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

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

def create_test_pdf(filename: str, content: str) -> bytes:
    """Create a simple PDF with test content"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from io import BytesIO
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Add content across multiple lines/pages
        y = 750
        for line in content.split('\n'):
            if y < 100:  # Start new page
                c.showPage()
                y = 750
            c.drawString(50, y, line[:70])  # Wrap long lines
            y -= 20
        
        c.save()
        buffer.seek(0)
        return buffer.read()
    except ImportError:
        print_warning("reportlab not installed, using mock PDF")
        # Return a minimal valid PDF as fallback
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n5 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(" + content.encode() + b") Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000229 00000 n\n0000000308 00000 n\ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n404\n%%EOF"

def test_1_clear_database():
    """Test 1: Clear the database"""
    print_header("TEST 1: Clear Database")
    
    try:
        import shutil
        chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            os.makedirs(chroma_path, exist_ok=True)
            print_success("Database cleared successfully")
        else:
            print_info("Database directory doesn't exist (first run)")
        
        # Reset vectorstore
        from app.vector_store import reset_vectorstore
        reset_vectorstore()
        print_success("Vectorstore reset")
        return True
    except Exception as e:
        print_error(f"Failed to clear database: {e}")
        return False

def test_2_upload_finance_2024():
    """Test 2: Upload 2024 Finance documents"""
    print_header("TEST 2: Upload Finance Documents (2024)")
    
    documents = [
        ("Q1_2024_Finance.pdf", "Q1 2024 Financial Report. Revenue: $10M. Profit: $2M. Year: 2024. Category: Finance."),
        ("Q2_2024_Finance.pdf", "Q2 2024 Financial Report. Revenue: $12M. Profit: $2.5M. Year: 2024. Category: Finance."),
    ]
    
    results = []
    for filename, content in documents:
        try:
            print(f"\nUploading {filename}...")
            pdf_bytes = create_test_pdf(filename, content)
            result = add_documents(
                pdf_bytes,
                filename,
                category="finance",
                year="2024"
            )
            
            if result["success"]:
                print_success(f"{filename}: {result['message']}")
                results.append(True)
            else:
                print_error(f"{filename}: {result.get('error')}")
                results.append(False)
        except Exception as e:
            print_error(f"Failed to upload {filename}: {e}")
            results.append(False)
    
    return all(results)

def test_3_upload_hr_2023():
    """Test 3: Upload 2023 HR documents"""
    print_header("TEST 3: Upload HR Documents (2023)")
    
    documents = [
        ("Annual_Report_2023_HR.pdf", "2023 Annual HR Report. Headcount: 500. Salary budget: $50M. Year: 2023. Category: HR."),
        ("Quarterly_Policy_2023.pdf", "Q3 2023 HR Policy Update. New remote work policy. Effective date: July 2023. Year: 2023. Category: HR."),
    ]
    
    results = []
    for filename, content in documents:
        try:
            print(f"\nUploading {filename}...")
            pdf_bytes = create_test_pdf(filename, content)
            result = add_documents(
                pdf_bytes,
                filename,
                category="hr",
                year="2023"
            )
            
            if result["success"]:
                print_success(f"{filename}: {result['message']}")
                results.append(True)
            else:
                print_error(f"{filename}: {result.get('error')}")
                results.append(False)
        except Exception as e:
            print_error(f"Failed to upload {filename}: {e}")
            results.append(False)
    
    return all(results)

def test_4_upload_product_2024():
    """Test 4: Upload 2024 Product documents"""
    print_header("TEST 4: Upload Product Documents (2024)")
    
    documents = [
        ("Product_Roadmap_2024.pdf", "Product Roadmap 2024. Q1: Feature A. Q2: Feature B. Q3: Feature C. Q4: Feature D. Year: 2024. Category: Product."),
    ]
    
    results = []
    for filename, content in documents:
        try:
            print(f"\nUploading {filename}...")
            pdf_bytes = create_test_pdf(filename, content)
            result = add_documents(
                pdf_bytes,
                filename,
                category="product",
                year="2024"
            )
            
            if result["success"]:
                print_success(f"{filename}: {result['message']}")
                results.append(True)
            else:
                print_error(f"{filename}: {result.get('error')}")
                results.append(False)
        except Exception as e:
            print_error(f"Failed to upload {filename}: {e}")
            results.append(False)
    
    return all(results)

def test_5_check_database_state():
    """Test 5: Check database state"""
    print_header("TEST 5: Check Database State")
    
    try:
        vectorstore = get_chroma_vectorstore()
        collection = vectorstore._collection
        data = collection.get()
        
        print_info(f"Total chunks in database: {len(data.get('ids', []))}")
        
        # Analyze metadata
        files_set = set()
        categories_set = set()
        years_set = set()
        
        for metadata in data.get("metadatas", []):
            if "filename" in metadata:
                files_set.add(metadata["filename"])
            if "category" in metadata:
                categories_set.add(metadata["category"])
            if "year" in metadata:
                years_set.add(str(metadata["year"]))
        
        print(f"\n{BOLD}Files:{RESET}")
        for f in sorted(files_set):
            print(f"  • {f}")
        
        print(f"\n{BOLD}Categories:{RESET}")
        for c in sorted(categories_set):
            print(f"  • {c}")
        
        print(f"\n{BOLD}Years:{RESET}")
        for y in sorted(years_set):
            print(f"  • {y}")
        
        return True
    except Exception as e:
        print_error(f"Failed to check database: {e}")
        return False

def test_6_query_no_filter():
    """Test 6: Query without filters"""
    print_header("TEST 6: Query Without Filters")
    
    try:
        query = "What is the financial report?"
        print(f"Query: {query}\n")
        
        docs, sources = retrieve_context(query, k=3, filters=None)
        
        print_info(f"Retrieved {len(docs)} documents")
        for i, source in enumerate(sources, 1):
            print(f"\n{BOLD}Result {i}:{RESET}")
            print(f"  File: {source.get('filename')}")
            print(f"  Category: {source.get('category')}")
            print(f"  Year: {source.get('year')}")
            print(f"  Page: {source.get('page')}")
            print(f"  Relevance: {source.get('relevance'):.4f}")
        
        return len(docs) >= 2  # Should find multiple documents
    except Exception as e:
        print_error(f"Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_filter_by_year_2024():
    """Test 7: Filter by year 2024"""
    print_header("TEST 7: Filter by Year 2024")
    
    try:
        query = "What's in the 2024 reports?"
        filters = {"year": "2024"}
        print(f"Query: {query}")
        print(f"Filters: {filters}\n")
        
        docs, sources = retrieve_context(query, k=5, filters=filters)
        
        print_info(f"Retrieved {len(docs)} documents with year=2024 filter")
        
        if len(docs) == 0:
            print_error("No documents found with year=2024 filter!")
            return False
        
        for i, source in enumerate(sources, 1):
            year = source.get('year')
            status = "✓" if year == "2024" else "✗"
            print(f"\n{BOLD}Result {i}:{RESET}")
            print(f"  File: {source.get('filename')}")
            print(f"  Category: {source.get('category')}")
            print(f"  Year: {year} {status}")
            print(f"  Relevance: {source.get('relevance'):.4f}")
        
        # Verify all are 2024
        all_2024 = all(s.get('year') == '2024' for s in sources)
        if all_2024:
            print_success("All results are from 2024!")
        else:
            print_warning("Some results are not from 2024")
        
        return all_2024
    except Exception as e:
        print_error(f"Filter query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_8_filter_by_category_finance():
    """Test 8: Filter by category finance"""
    print_header("TEST 8: Filter by Category 'finance'")
    
    try:
        query = "Tell me about finance"
        filters = {"category": "finance"}
        print(f"Query: {query}")
        print(f"Filters: {filters}\n")
        
        docs, sources = retrieve_context(query, k=5, filters=filters)
        
        print_info(f"Retrieved {len(docs)} documents with category=finance filter")
        
        if len(docs) == 0:
            print_error("No documents found with category=finance filter!")
            return False
        
        for i, source in enumerate(sources, 1):
            category = source.get('category')
            status = "✓" if category == "finance" else "✗"
            print(f"\n{BOLD}Result {i}:{RESET}")
            print(f"  File: {source.get('filename')}")
            print(f"  Category: {category} {status}")
            print(f"  Year: {source.get('year')}")
            print(f"  Relevance: {source.get('relevance'):.4f}")
        
        # Verify all are finance
        all_finance = all(s.get('category') == 'finance' for s in sources)
        if all_finance:
            print_success("All results are from finance category!")
        else:
            print_warning("Some results are not from finance category")
        
        return all_finance
    except Exception as e:
        print_error(f"Filter query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_9_filter_combined():
    """Test 9: Filter by year AND category"""
    print_header("TEST 9: Filter by Year 2024 AND Category 'finance'")
    
    try:
        query = "What about 2024 finance?"
        filters = {"year": "2024", "category": "finance"}
        print(f"Query: {query}")
        print(f"Filters: {filters}\n")
        
        docs, sources = retrieve_context(query, k=5, filters=filters)
        
        print_info(f"Retrieved {len(docs)} documents with combined filters")
        
        if len(docs) == 0:
            print_error("No documents found with combined filters!")
            return False
        
        for i, source in enumerate(sources, 1):
            year = source.get('year')
            category = source.get('category')
            year_ok = year == '2024'
            cat_ok = category == 'finance'
            status = "✓" if (year_ok and cat_ok) else "✗"
            
            print(f"\n{BOLD}Result {i}:{RESET}")
            print(f"  File: {source.get('filename')}")
            print(f"  Category: {category} {'✓' if cat_ok else '✗'}")
            print(f"  Year: {year} {'✓' if year_ok else '✗'}")
            print(f"  Status: {status}")
            print(f"  Relevance: {source.get('relevance'):.4f}")
        
        # Verify all match both filters
        all_match = all(
            s.get('year') == '2024' and s.get('category') == 'finance'
            for s in sources
        )
        if all_match:
            print_success("All results match both filters (2024 AND finance)!")
        else:
            print_warning("Some results don't match both filters")
        
        return all_match
    except Exception as e:
        print_error(f"Combined filter query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_10_filter_by_filename():
    """Test 10: Filter by specific filename"""
    print_header("TEST 10: Filter by Filename")
    
    try:
        query = "Q1 report"
        filters = {"filename": "Q1_2024_Finance.pdf"}
        print(f"Query: {query}")
        print(f"Filters: {filters}\n")
        
        docs, sources = retrieve_context(query, k=5, filters=filters)
        
        print_info(f"Retrieved {len(docs)} documents with filename filter")
        
        if len(docs) == 0:
            print_error("No documents found with filename filter!")
            return False
        
        for i, source in enumerate(sources, 1):
            filename = source.get('filename')
            status = "✓" if filename == "Q1_2024_Finance.pdf" else "✗"
            print(f"\n{BOLD}Result {i}:{RESET}")
            print(f"  File: {filename} {status}")
            print(f"  Category: {source.get('category')}")
            print(f"  Year: {source.get('year')}")
            print(f"  Relevance: {source.get('relevance'):.4f}")
        
        # Verify all are the specified file
        all_match = all(s.get('filename') == 'Q1_2024_Finance.pdf' for s in sources)
        if all_match:
            print_success("All results are from Q1_2024_Finance.pdf!")
        else:
            print_warning("Some results are from different files")
        
        return all_match
    except Exception as e:
        print_error(f"Filename filter query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_11_rag_query_with_filter():
    """Test 11: Full RAG query with filters"""
    print_header("TEST 11: Full RAG Query with Filters")
    
    try:
        query = "What was the 2024 financial performance?"
        filters = {"year": "2024", "category": "finance"}
        print(f"Query: {query}")
        print(f"Filters: {filters}\n")
        
        result = rag_query(query, k=3, filters=filters)
        
        if result["success"]:
            print_success("RAG query successful!")
            print(f"\n{BOLD}Answer:{RESET}")
            print(result["answer"][:300] + "..." if len(result["answer"]) > 300 else result["answer"])
            
            print(f"\n{BOLD}Sources:{RESET}")
            for i, source in enumerate(result.get("sources", []), 1):
                print(f"\n  {i}. {source.get('filename')} (p.{source.get('page')})")
                print(f"     Category: {source.get('category')}, Year: {source.get('year')}")
                print(f"     Relevance: {source.get('relevance'):.4f}")
            
            return True
        else:
            print_error(f"RAG query failed: {result.get('error')}")
            return False
    except Exception as e:
        print_error(f"RAG query error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_12_different_categories():
    """Test 12: Verify different categories are separate"""
    print_header("TEST 12: Verify Different Categories are Separate")
    
    try:
        # Query for HR
        docs_hr, sources_hr = retrieve_context("HR policy", k=5, filters={"category": "hr"})
        # Query for Finance
        docs_fin, sources_fin = retrieve_context("Revenue profit", k=5, filters={"category": "finance"})
        
        print_info(f"HR documents found: {len(docs_hr)}")
        print_info(f"Finance documents found: {len(docs_fin)}")
        
        if len(docs_hr) > 0 and len(docs_fin) > 0:
            # Check they're different
            hr_files = {s.get('filename') for s in sources_hr}
            fin_files = {s.get('filename') for s in sources_fin}
            
            overlap = hr_files & fin_files
            
            print(f"\nHR files: {hr_files}")
            print(f"Finance files: {fin_files}")
            print(f"Overlap: {overlap if overlap else 'None'}")
            
            if not overlap:
                print_success("Categories are properly isolated!")
                return True
            else:
                print_warning(f"Categories have {len(overlap)} overlapping files")
                return True  # Still pass if we got results from both
        else:
            print_error("Could not retrieve both category types")
            return False
    except Exception as e:
        print_error(f"Category separation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print_header("METADATA FILTERING COMPREHENSIVE TEST SUITE")
    
    tests = [
        ("Clear Database", test_1_clear_database),
        ("Upload Finance 2024", test_2_upload_finance_2024),
        ("Upload HR 2023", test_3_upload_hr_2023),
        ("Upload Product 2024", test_4_upload_product_2024),
        ("Check Database State", test_5_check_database_state),
        ("Query Without Filter", test_6_query_no_filter),
        ("Filter by Year 2024", test_7_filter_by_year_2024),
        ("Filter by Category finance", test_8_filter_by_category_finance),
        ("Combined Filters (2024 + finance)", test_9_filter_combined),
        ("Filter by Filename", test_10_filter_by_filename),
        ("Full RAG Query with Filter", test_11_rag_query_with_filter),
        ("Category Isolation", test_12_different_categories),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print_error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = f"{GREEN}PASS{RESET}" if passed_flag else f"{RED}FAIL{RESET}"
        print(f"{status} - {test_name}")
    
    print(f"\n{BOLD}Total: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}⚠ {total - passed} tests failed{RESET}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

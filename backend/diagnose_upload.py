#!/usr/bin/env python
"""Diagnostic script to test PDF upload"""
import requests
import json
import sys

print("=" * 60)
print("RAG UPLOAD DIAGNOSTIC TEST")
print("=" * 60)

API_URL = "http://127.0.0.1:8000"

# Test 1: Check if backend is running
print("\n[TEST 1] Backend Health Check")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✓ Backend is running")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Backend returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Cannot connect to backend: {e}")
    print(f"  Make sure backend is running on {API_URL}")
    sys.exit(1)

# Test 2: Create a simple PDF
print("\n[TEST 2] Creating test PDF")
pdf_data = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000056 00000 n
0000000115 00000 n
0000000244 00000 n
trailer<</Size 5/Root 1 0 R>>
startxref
338
%%EOF"""

print(f"✓ Test PDF created ({len(pdf_data)} bytes)")

# Test 3: Upload PDF
print("\n[TEST 3] Uploading PDF to /upload endpoint")
try:
    files = {'file': ('test.pdf', pdf_data, 'application/pdf')}
    print(f"  Sending POST request to {API_URL}/upload")
    
    response = requests.post(
        f"{API_URL}/upload",
        files=files,
        timeout=30
    )
    
    print(f"  Status Code: {response.status_code}")
    print(f"  Response Headers: {dict(response.headers)}")
    print(f"  Response Body:\n{json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print(f"\n✓ UPLOAD SUCCESSFUL!")
            print(f"  Chunks created: {result.get('chunks_count')}")
        else:
            print(f"\n✗ Upload failed: {result.get('error')}")
    else:
        print(f"\n✗ Upload failed with status {response.status_code}")
        
except Exception as e:
    print(f"✗ Error during upload: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

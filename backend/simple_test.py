#!/usr/bin/env python
"""Simple test to see upload response"""
import json

# Create a minimal valid PDF
pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000102 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
149
%%EOF"""

print(f"PDF content size: {len(pdf_content)} bytes")
print("Attempting to upload...")

try:
    import requests
    files = {'file': ('test.pdf', pdf_content, 'application/pdf')}
    response = requests.post('http://127.0.0.1:8000/upload', files=files, timeout=60)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Response Text: {response.text[:500]}")
    
    try:
        data = response.json()
        print(f"\nJSON Response: {json.dumps(data, indent=2)}")
    except:
        print("Response is not JSON")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

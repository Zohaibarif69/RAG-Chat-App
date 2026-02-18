#!/usr/bin/env python3
import sys
print(f"Python: {sys.executable}")

print("\n1. Testing sentence_transformers import...")
try:
    import sentence_transformers
    print(f"✓ sentence_transformers imported from: {sentence_transformers.__file__}")
except Exception as e:
    print(f"✗ Error importing sentence_transformers: {e}")
    sys.exit(1)

print("\n2. Testing HuggingFaceEmbeddings import...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✓ HuggingFaceEmbeddings imported")
except Exception as e:
    print(f"✗ Error importing HuggingFaceEmbeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing model initialization...")
try:
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ ALL TESTS PASSED")

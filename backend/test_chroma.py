import sys
import os

# Suppress progress bars to speed up loading
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Starting test...", file=sys.stderr)
print("Loading sentence-transformers model (this is slow on first run)...", file=sys.stderr)

try:
    from app.vector_store import get_vectorstore
    print("✓ Imported get_vectorstore", file=sys.stderr)

    vectorstore = get_vectorstore()
    print("✓ Got vectorstore", file=sys.stderr)

    vectorstore.add_texts([
        "FastAPI is a Python web framework.",
        "Chroma stores embeddings for retrieval.",
        "Next.js is used for frontend development."
    ])

    print("\n✓ Inserted successfully!")

    results = vectorstore.similarity_search("What is FastAPI?")
    print(f"\n✓ Search results: {results}")
    print(f"✓ Number of results: {len(results)}")
    print("\n=== CHROMA TEST SUCCESS ===")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()

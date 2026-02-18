from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from .vector_store import add_documents, rag_query, get_all_documents, delete_document

# Load environment variables
load_dotenv()

# Debug: Check if API key is loaded
import sys
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found in environment", file=sys.stderr)
else:
    print("INFO: GROQ_API_KEY loaded successfully", file=sys.stderr)

app = FastAPI(title="RAG Chat API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    success: bool
    answer: str = None
    sources: list = None
    error: str = None


# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "yes it is running", "version": "1.0"}

@app.post("/clear-db")
def clear_database():
    """Clear the vector database"""
    try:
        import shutil
        import os
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        chroma_path = os.path.abspath(chroma_path)
        
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            os.makedirs(chroma_path, exist_ok=True)
        
        # Reset the global vectorstore variable
        from .vector_store import reset_vectorstore
        reset_vectorstore()
        
        return {
            "success": True,
            "message": "Database cleared successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
def health():
    """Detailed health check"""
    try:
        # Try to initialize embedding to see if that's the issue
        from .vector_store import get_embedding
        embedding = get_embedding()
        return {
            "status": "healthy",
            "embedding_model": "loaded"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Upload PDF endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF file and add to vector store"""
    try:
        print(f"Received upload request for: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        if file.content_type != "application/pdf":
            print(f"Wrong content type: {file.content_type}")
            return {
                "success": False,
                "error": f"File must be a PDF, got {file.content_type}"
            }
        
        # Read PDF content
        print("Reading file content...")
        content = await file.read()
        print(f"File size: {len(content)} bytes")
        
        if len(content) == 0:
            return {
                "success": False,
                "error": "Uploaded file is empty"
            }
        
        # Process and add to vector store
        print("Processing PDF...")
        result = add_documents(content, file.filename)
        print(f"Processing result: {result}")
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "chunks_count": result["chunks_count"]
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error during processing")
            }
            
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"ERROR in upload: {error_msg}")
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Upload failed: {error_msg}"
        }


# RAG Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG system and get answer with sources"""
    try:
        result = rag_query(request.question, request.top_k)
        
        if result["success"]:
            return QueryResponse(
                success=True,
                answer=result["answer"],
                sources=result["sources"]
            )
        else:
            return QueryResponse(
                success=False,
                error=result["error"]
            )
            
    except Exception as e:
        return QueryResponse(
            success=False,
            error=str(e)
        )


# Get all documents endpoint
@app.get("/documents")
def list_documents():
    """Get list of all uploaded documents"""
    try:
        documents = get_all_documents()
        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Delete document endpoint
@app.post("/delete-document")
def delete_doc(source_name: str):
    """Delete a specific document by name"""
    try:
        result = delete_document(source_name)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

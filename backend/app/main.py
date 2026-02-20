from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
from typing import Optional, List
from .vector_store import add_documents, rag_query, get_all_documents, delete_document
from .database import init_db, create_session, get_session, save_message, get_messages, update_session_documents
from .evaluation import get_evaluator, QueryMetricsTracker
import uuid

# Load environment variables
load_dotenv()

# Initialize database
init_db()

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
    session_id: str
    top_k: int = 5


class QueryResponse(BaseModel):
    success: bool
    answer: str = None
    sources: list = None
    error: str = None


class ChatSessionRequest(BaseModel):
    session_id: Optional[str] = None


class ChatSessionResponse(BaseModel):
    session_id: str
    created_at: str


class MessageResponse(BaseModel):
    role: str
    content: str
    sources: Optional[list] = None
    created_at: str


class ChatHistoryResponse(BaseModel):
    success: bool
    messages: List[MessageResponse] = []
    error: Optional[str] = None


# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "yes it is running", "version": "1.0"}


# Chat session endpoints
@app.post("/session", response_model=ChatSessionResponse)
def create_new_session(request: ChatSessionRequest):
    """Create a new chat session"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        existing_session = get_session(session_id)
        
        if existing_session:
            return ChatSessionResponse(
                session_id=session_id,
                created_at=existing_session.created_at.isoformat()
            )
        
        session = create_session(session_id)
        return ChatSessionResponse(
            session_id=session.id,
            created_at=session.created_at.isoformat()
        )
    except Exception as e:
        print(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = get_messages(session_id)
        return ChatHistoryResponse(
            success=True,
            messages=[
                MessageResponse(
                    role=msg.role,
                    content=msg.content,
                    sources=json.loads(msg.sources) if msg.sources else None,
                    created_at=msg.created_at.isoformat()
                )
                for msg in messages
            ]
        )
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return ChatHistoryResponse(
            success=False,
            error=str(e)
        )


@app.post("/session/{session_id}/documents")
def update_session_docs(session_id: str, document_names: str):
    """Update documents for a session"""
    try:
        update_session_documents(session_id, document_names)
        return {"success": True, "message": "Documents updated"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    """Simple health check"""
    return {
        "status": "healthy",
        "message": "Backend is running"
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


# RAG Query endpoint with conversation history and evaluation
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG system with multi-turn conversation support and metrics tracking"""
    try:
        # Initialize evaluation tracking
        evaluator = get_evaluator()
        metrics_tracker = QueryMetricsTracker(request.question, evaluator)
        
        # Get conversation history (last 5 messages)
        metrics_tracker.start_phase("conversation_history")
        history = get_messages(request.session_id, limit=10)
        metrics_tracker.end_phase()
        
        # Build conversation context
        conversation_context = ""
        if history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in history[-5:]:  # Last 5 messages for context
                role = "User" if msg.role == "user" else "Assistant"
                conversation_context += f"{role}: {msg.content}\n"
        
        # Add question with context
        full_question = request.question
        if conversation_context:
            full_question = f"{conversation_context}\nUser: {request.question}"
        
        # Get RAG answer
        metrics_tracker.start_phase("rag_query_execution")
        result = rag_query(full_question, request.top_k)
        metrics_tracker.end_phase()
        
        if result["success"]:
            # Extract document sources for logging
            doc_sources = [source.get("source", "Unknown") for source in result.get("sources", [])]
            
            # Log the query execution
            metrics_tracker.start_phase("database_save")
            save_message(request.session_id, "user", request.question)
            save_message(request.session_id, "assistant", result["answer"], json.dumps(result.get("sources", [])))
            metrics_tracker.end_phase()
            
            # Record metrics
            _ = metrics_tracker.log_result(
                retrieved_docs=doc_sources,
                final_answer=result["answer"],
                metadata={
                    "reasoning_method": result.get("reasoning_method", "hybrid"),
                    "reflection": result.get("reflection", {}),
                    "session_id": request.session_id
                }
            )
            
            return QueryResponse(
                success=True,
                answer=result["answer"],
                sources=result["sources"]
            )
        else:
            evaluator.metrics["failed_queries"] += 1
            return QueryResponse(
                success=False,
                error=result["error"]
            )
            
    except Exception as e:
        print(f"Error in query: {e}")
        import traceback
        traceback.print_exc()
        evaluator = get_evaluator()
        evaluator.metrics["failed_queries"] += 1
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


# Evaluation & Metrics endpoints
@app.get("/metrics")
def get_metrics():
    """Get RAG system evaluation metrics"""
    try:
        evaluator = get_evaluator()
        summary = evaluator.get_metrics_summary()
        return {
            "success": True,
            "metrics": summary
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/logs/export")
def export_logs():
    """Export all logs to file"""
    try:
        evaluator = get_evaluator()
        filepath = evaluator.save_logs()
        return {
            "success": True,
            "message": f"Logs exported to {filepath}",
            "filepath": filepath
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/metrics/summary")
def get_summary():
    """Get concise metrics summary"""
    try:
        evaluator = get_evaluator()
        summary = evaluator.get_metrics_summary()
        
        return {
            "success": True,
            "summary": {
                "total_queries": summary.get("total_queries", 0),
                "success_rate": f"{summary.get('success_rate', 0):.1%}",
                "avg_latency_ms": f"{summary.get('avg_latency_ms', 0):.0f}",
                "avg_cost_usd": f"${summary.get('avg_cost_usd', 0):.4f}",
                "hallucination_rate": f"{summary.get('hallucination_rate', 0):.1%}",
                "avg_precision": f"{summary.get('avg_retrieval_precision', 0):.3f}",
                "avg_recall": f"{summary.get('avg_retrieval_recall', 0):.3f}"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

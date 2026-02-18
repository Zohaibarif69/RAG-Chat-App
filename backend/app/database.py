"""
Database setup and models for chat persistence
"""
import os
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Optional, List

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:zohaib123@localhost:5432/rag_db")

# Create database engine
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class ChatSession(Base):
    """Chat session model - represents a conversation"""
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, index=True)  # UUID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    title = Column(String(255), nullable=True)
    document_names = Column(Text, nullable=True)  # comma-separated list of uploaded PDFs


class ChatMessage(Base):
    """Message model - represents a single message in a conversation"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role = Column(String(10), index=True)  # "user" or "assistant"
    content = Column(Text)
    sources = Column(Text, nullable=True)  # JSON string of sources
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db():
    """Initialize database tables"""
    print("Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_session(session_id: str, document_names: Optional[str] = None) -> ChatSession:
    """Create a new chat session"""
    db = SessionLocal()
    try:
        chat_session = ChatSession(
            id=session_id,
            document_names=document_names
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        return chat_session
    finally:
        db.close()


def get_session(session_id: str) -> Optional[ChatSession]:
    """Get a chat session by ID"""
    db = SessionLocal()
    try:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()
    finally:
        db.close()


def save_message(session_id: str, role: str, content: str, sources: Optional[str] = None) -> ChatMessage:
    """Save a message to the database"""
    db = SessionLocal()
    try:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message
    finally:
        db.close()


def get_messages(session_id: str, limit: int = 20) -> List[ChatMessage]:
    """Get last N messages from a session"""
    db = SessionLocal()
    try:
        return db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.created_at.desc())\
            .limit(limit)\
            .all()[::-1]  # Reverse to get chronological order
    finally:
        db.close()


def update_session_documents(session_id: str, document_names: str):
    """Update the list of documents for a session"""
    db = SessionLocal()
    try:
        db.query(ChatSession).filter(ChatSession.id == session_id).update(
            {"document_names": document_names, "updated_at": datetime.utcnow()}
        )
        db.commit()
    finally:
        db.close()

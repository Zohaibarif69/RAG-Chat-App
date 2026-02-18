#!/usr/bin/env python3
"""
Initialize the chat memory database
"""
import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set up environment
os.environ['TZ'] = 'UTC'

try:
    from app.database import init_db
    
    print("Initializing chat memory database...")
    init_db()
    print(" Database initialized successfully!")
    print("\nDatabase tables created:")
    print("  - chat_sessions: Stores conversation sessions")
    print("  - chat_messages: Stores user and assistant messages")
    
except Exception as e:
    print(f" Error initializing database: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

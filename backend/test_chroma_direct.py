#!/usr/bin/env python
"""Direct test of Chroma database contents"""

import sqlite3
import os

chroma_db_path = os.path.join(os.path.dirname(__file__), "chroma_db", "chroma.sqlite3")

print(f"Checking Chroma database at: {chroma_db_path}")
print(f"File exists: {os.path.exists(chroma_db_path)}")

if os.path.exists(chroma_db_path):
    conn = sqlite3.connect(chroma_db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"\nTables in database: {tables}")
    
    # Query embeddings table if it exists
    try:
        cursor.execute("SELECT COUNT(*) FROM embeddings;")
        count = cursor.fetchone()[0]
        print(f"Total embeddings: {count}")
        
        if count > 0:
            cursor.execute("SELECT id, document FROM embeddings LIMIT 5;")
            rows = cursor.fetchall()
            print(f"Sample embeddings: {rows}")
    except Exception as e:
        print(f"Error querying embeddings: {e}")
    
    # Query documents table if it exists
    try:
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()[0]
        print(f"Total documents: {count}")
        
        if count > 0:
            cursor.execute("SELECT id, metadata FROM documents LIMIT 5;")
            rows = cursor.fetchall()
            print(f"Sample documents: {rows}")
    except Exception as e:
        print(f"Error querying documents: {e}")
    
    # Try to list all tables and their row counts
    print("\n=== All tables and their row counts ===")
    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"{table_name}: {count} rows")
        except:
            pass
    
    conn.close()
else:
    print("Chroma database file not found!")

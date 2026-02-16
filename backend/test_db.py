import os
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

print("=" * 50)
print("Testing Environment Variables")
print("=" * 50)

# Check if .env file exists
if os.path.exists('.env'):
    print(".env file found")
else:
    print(" .env file not found")

# Load DATABASE_URL
database_url = os.getenv('DATABASE_URL')

print("\nEnvironment Variables Loaded:")
print(f"  DATABASE_URL: {database_url if database_url else 'Not set'}")

if not database_url:
    print("\n DATABASE_URL not found in .env file")
else:
    print(" DATABASE_URL is set")

print("\n" + "=" * 50)
print("Testing PostgreSQL Connection")
print("=" * 50)

try:
    if not database_url:
        print(" Cannot test PostgreSQL connection - DATABASE_URL not set")
    else:
        connection = psycopg2.connect(database_url)
        
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        print(" Successfully connected to PostgreSQL")
        print(f"  Database version: {version[0]}")
        
        cursor.close()
        connection.close()
        
except Exception as e:
    print(f" Failed to connect to PostgreSQL: {str(e)}")

print("\n" + "=" * 50)
print("Test Complete")
print("=" * 50)

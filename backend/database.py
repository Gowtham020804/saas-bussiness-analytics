import sqlite3
import os

# Dynamically map the users.db relative to backend/ directory (to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_db_connection():
    """Establishes SQLite connection to users.db"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

# Initialize tables
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()
conn.close()

import jwt
import bcrypt
from datetime import datetime, timedelta, timezone

JWT_SECRET = "saas_analytics_super_secret_key_123!"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

def hash_password(password: str) -> str:
    """Hashes a raw password using native bcrypt library"""
    # bcrypt expects bytes payload
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password_bytes, salt)
    # Return as string for database storage
    return hashed_bytes.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain password against the hashed hash.
    Safely supports legacy plain-text passwords for backward compatibility.
    """
    try:
        # Check if the database entry looks like a valid bcrypt hash
        if not (hashed_password.startswith("$2a$") or hashed_password.startswith("$2b$") or hashed_password.startswith("$2y$")):
            # Legacy plain-text password support
            return plain_password == hashed_password
            
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        # Fallback comparison in case of format exceptions
        return plain_password == hashed_password

def generate_jwt_token(user_id: int, name: str, email: str) -> str:
    """Generates a secure JSON Web Token valid for 24 hours"""
    expiration = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_jwt_token(token: str) -> dict:
    """Decodes and validates a JWT token. Returns payload or None if invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

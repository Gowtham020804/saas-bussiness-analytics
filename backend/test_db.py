import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_connection
from backend.auth import hash_password, verify_password, generate_jwt_token

try:
    print("Testing connection...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    print("Found users:", users)
    
    # Try hashing / verifying password
    h = hash_password("test")
    print("Hash:", h)
    v = verify_password("test", h)
    print("Verify test:", v)
    
    # Test with first user if exists
    if len(users) > 0:
        db_user = users[0]
        print("Testing verification of user:", db_user)
        try:
            # This will fail if the stored password is plain text
            v_db = verify_password("test", db_user[3])
            print("Verify DB:", v_db)
        except Exception as e:
            print("Verify DB password failed with exception:", type(e), e)
            
    conn.close()
    print("All tests done!")
except Exception as e:
    import traceback
    print("TEST FAILED:")
    traceback.print_exc()

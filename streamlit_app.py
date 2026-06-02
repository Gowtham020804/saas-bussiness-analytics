import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import requests
import datetime
import time
import os

# =========================================================
# CONFIGURATION & API PATHS
# =========================================================
API_BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="SaaS Analytics & Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# BACKEND ALIVE VERIFICATION & GOOGLE OAUTH HELPERS
# =========================================================
def check_backend_alive():
    """Verifies if the FastAPI server is reachable"""
    try:
        response = requests.get(f"http://127.0.0.1:8000/docs", timeout=1.5)
        return True
    except:
        return False

def load_env_file():
    env_vars = {}
    paths = [".env", "backend/.env"]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        try:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip().strip('"').strip("'")
                        except Exception:
                            pass
    return env_vars

def get_google_credentials():
    # 1. Try Streamlit Secrets
    try:
        client_id = st.secrets.get("GOOGLE_CLIENT_ID", "")
        client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
        redirect_uri = st.secrets.get("GOOGLE_REDIRECT_URI", "")
    except Exception:
        client_id = ""
        client_secret = ""
        redirect_uri = ""
    
    # 2. Try environment variables
    if not client_id:
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    if not client_secret:
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if not redirect_uri:
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "")
        
    # 3. Try .env file
    if not client_id or not client_secret:
        env_vars = load_env_file()
        if not client_id:
            client_id = env_vars.get("GOOGLE_CLIENT_ID", "")
        if not client_secret:
            client_secret = env_vars.get("GOOGLE_CLIENT_SECRET", "")
        if not redirect_uri:
            redirect_uri = env_vars.get("GOOGLE_REDIRECT_URI", "")
            
    # 4. Try session state
    if not client_id:
        client_id = st.session_state.get("google_client_id", "")
    if not client_secret:
        client_secret = st.session_state.get("google_client_secret", "")
    if not redirect_uri:
        redirect_uri = st.session_state.get("google_redirect_uri", "http://localhost:8501/")
        
    return client_id, client_secret, redirect_uri

def save_google_credentials(client_id, client_secret, redirect_uri):
    st.session_state.google_client_id = client_id
    st.session_state.google_client_secret = client_secret
    st.session_state.google_redirect_uri = redirect_uri
    
    # Save to local .env
    existing_vars = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s and not line_s.startswith("#") and "=" in line_s:
                        try:
                            k, v = line_s.split("=", 1)
                            existing_vars[k.strip()] = line_s
                        except:
                            pass
        except Exception:
            pass
            
    existing_vars["GOOGLE_CLIENT_ID"] = f"GOOGLE_CLIENT_ID={client_id}"
    existing_vars["GOOGLE_CLIENT_SECRET"] = f"GOOGLE_CLIENT_SECRET={client_secret}"
    existing_vars["GOOGLE_REDIRECT_URI"] = f"GOOGLE_REDIRECT_URI={redirect_uri}"
    
    try:
        with open(".env", "w", encoding="utf-8") as f:
            for k, line_content in existing_vars.items():
                f.write(line_content + "\n")
    except Exception as e:
        st.warning(f"Could not save credentials to local .env file: {e}")

# Setup dynamic Mode selector
is_backend_alive = check_backend_alive()
if not is_backend_alive:
    # RUNNING IN SELF-CONTAINED LOCAL FALLBACK MODE
    # Import backend logic locally to run calculations inside the Streamlit process
    import sqlite3
    from backend.auth import hash_password, verify_password, generate_jwt_token
    from backend.upload import generate_sample_saas_data
    from backend.analytics import preprocess_dataframe, compute_saas_kpis, get_charts_data, get_cohort_matrix_data
    from backend.ml.churn_model import train_churn_model, get_active_ledger_predictions, simulate_customer_churn_risk

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "df_details" not in st.session_state:
    st.session_state.df_details = None  # Stores {filename, rows, columns, preview_json}
if "mapped_columns" not in st.session_state:
    st.session_state.mapped_columns = {}
if "show_google_modal" not in st.session_state:
    st.session_state.show_google_modal = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}
if "feature_importances" not in st.session_state:
    st.session_state.feature_importances = []

# Fallback Cache for Local Mode Dataframes & ML Models (stored in session state)
if "local_df" not in st.session_state:
    st.session_state.local_df = None

# =========================================================
# THEME & CUSTOM STYLING (PREMIUM DARK GLASSMORPHISM)
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Base Overrides */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #060913 !important;
        color: #E2E8F0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0A0E1A !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Radial Glow Effects */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        width: 800px;
        height: 800px;
        top: -200px;
        left: -200px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, rgba(0, 229, 255, 0.02) 50%, rgba(0,0,0,0) 100%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Custom Glassmorphic Containers */
    .premium-card {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease !important;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 229, 255, 0.3) !important;
    }
    
    /* Glowing Stat Indicators */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    /* Buttons Override */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #00E5FF 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        width: 100%;
        text-align: center;
        height: auto !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #00b8d4 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.4) !important;
    }
    
    /* Custom Titles */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        border-left: 4px solid #6366F1;
        padding-left: 12px;
        margin-bottom: 20px;
        margin-top: 15px;
    }
    
    /* Custom Alerts */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# API CORE HELPERS & LOCAL DB HELPERS
# =========================================================

def get_auth_headers():
    """Generates standard Bearer Token Authorization Headers"""
    token = st.session_state.jwt_token
    return {"Authorization": f"Bearer {token}"} if token else {}

def handle_api_error(response):
    """Parses and logs response errors from the REST API"""
    try:
        err_detail = response.json().get("detail", "Unknown server exception occurred.")
    except:
        err_detail = response.text or "Server returned empty error."
    st.error(f"❌ Server Error ({response.status_code}): {err_detail}")

# SQLite connection helper for local fallback mode
def get_local_db_conn():
    # Store database in working directory
    db_path = os.path.join(os.getcwd(), "users.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
    return conn

# Banner indicating current mode
if is_backend_alive:
    st.sidebar.success("⚡ Connected to FastAPI REST Backend")
else:
    st.sidebar.warning("⚡ Running in self-contained Local Mode")

# Initialize Google Client session state variables
if "google_client_id" not in st.session_state:
    st.session_state.google_client_id = ""
if "google_client_secret" not in st.session_state:
    st.session_state.google_client_secret = ""
if "google_redirect_uri" not in st.session_state:
    st.session_state.google_redirect_uri = "http://localhost:8501/"

# Process Google OAuth Authorization Code Callback
query_params = st.query_params
if "code" in query_params:
    auth_code = query_params["code"]
    c_id, c_secret, r_uri = get_google_credentials()
    
    if c_id and c_secret:
        # Perform Token Exchange
        token_url = "https://oauth2.googleapis.com/token"
        token_payload = {
            "code": auth_code,
            "client_id": c_id,
            "client_secret": c_secret,
            "redirect_uri": r_uri,
            "grant_type": "authorization_code"
        }
        try:
            # We must execute token request synchronously to block Streamlit execution
            res = requests.post(token_url, data=token_payload)
            if res.status_code == 200:
                token_data = res.json()
                access_token = token_data.get("access_token")
                
                # Retrieve User Info from Google
                userinfo_res = requests.get(
                    "https://www.googleapis.com/oauth2/v3/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if userinfo_res.status_code == 200:
                    user_info = userinfo_res.json()
                    email = user_info.get("email")
                    name = user_info.get("name", email.split("@")[0])
                    
                    if is_backend_alive:
                        # Log in on FastAPI Backend
                        payload = {"name": name, "email": email}
                        backend_res = requests.post(f"{API_BASE_URL}/auth/google", json=payload)
                        if backend_res.status_code == 200:
                            data = backend_res.json()
                            st.session_state.logged_in = True
                            st.session_state.jwt_token = data["access_token"]
                            st.session_state.user_name = data["user"]["name"]
                            st.session_state.user_email = data["user"]["email"]
                            st.session_state.show_google_modal = False
                            st.query_params.clear()
                            st.success("✅ Signed in successfully via Google!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ Backend Google Sign-in failed: {backend_res.text}")
                    else:
                        # Log in Locally
                        conn = get_local_db_conn()
                        cursor = conn.cursor()
                        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
                        db_user = cursor.fetchone()
                        if not db_user:
                            cursor.execute(
                                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                                (name, email, hash_password("google-oauth-simulated-password"))
                            )
                            conn.commit()
                            cursor.execute("SELECT * FROM users WHERE email=?", (email,))
                            db_user = cursor.fetchone()
                        conn.close()
                        
                        st.session_state.logged_in = True
                        st.session_state.jwt_token = generate_jwt_token(db_user[0], db_user[1], db_user[2])
                        st.session_state.user_name = db_user[1]
                        st.session_state.user_email = db_user[2]
                        st.session_state.show_google_modal = False
                        st.query_params.clear()
                        st.success("✅ Signed in successfully via Google!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.error("❌ Google profile retrieval failed.")
            else:
                st.error(f"❌ Google OAuth exchange failed: {res.text}")
        except Exception as ex:
            st.error(f"❌ Connection error during Google OAuth: {ex}")

# =========================================================
# GOOGLE SIGN-IN INTERACTIVE MODAL OVERLAY
# =========================================================
if st.session_state.show_google_modal:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px; padding-top: 10px;">
                <img src="https://www.vectorlogo.zone/logos/google/google-icon.svg" width="44" height="44" style="margin-bottom: 15px;"/>
                <h3 style="color: white; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 700; margin: 0;">Sign in with Google</h3>
                <p style="color: #94A3B8; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.85rem; margin-top: 5px;">Configure credentials to authenticate dynamically via Google OAuth 2.0</p>
            </div>
            <hr style="border-color: rgba(255,255,255,0.08); margin-bottom: 20px;" />
            """, unsafe_allow_html=True)
            
            client_id, client_secret, redirect_uri = get_google_credentials()
            
            # If credentials are not configured, show setup form
            if not client_id or not client_secret:
                st.warning("⚠️ Google OAuth credentials are not fully configured. Set them up below to enable actual Google Sign-In.")
                
                new_client_id = st.text_input("Google Client ID", value=client_id, placeholder="e.g. 123456-abcdef.apps.googleusercontent.com")
                new_client_secret = st.text_input("Google Client Secret", value=client_secret, type="password", placeholder="e.g. GOCSPX-xxxxxxxxx")
                new_redirect_uri = st.text_input("Authorized Redirect URI", value=redirect_uri, placeholder="e.g. http://localhost:8501/")
                
                st.markdown("""
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px; margin-bottom: 15px;">
                    <strong>Instructions:</strong><br>
                    1. Go to the <a href="https://console.cloud.google.com/" target="_blank" style="color: #6366F1;">Google Cloud Console</a>.<br>
                    2. Create an OAuth 2.0 Client ID (Web Application type).<br>
                    3. Add <code>http://localhost:8501/</code> (or your deployed app domain URL) as an <strong>Authorized Redirect URI</strong>.<br>
                    4. Copy the Client ID and Client Secret and paste them above.
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("💾 Save Credentials & Continue", use_container_width=True):
                    if new_client_id and new_client_secret and new_redirect_uri:
                        save_google_credentials(new_client_id, new_client_secret, new_redirect_uri)
                        st.success("✅ Credentials saved successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Please fill in all configuration fields.")
            else:
                # Credentials are set up! Show the authentic Google sign-in button
                google_auth_url = (
                    f"https://accounts.google.com/o/oauth2/v2/auth?"
                    f"client_id={client_id}&"
                    f"redirect_uri={redirect_uri}&"
                    f"response_type=code&"
                    f"scope=openid%20email%20profile&"
                    f"state=saas_analytics_auth"
                )
                
                st.write("🔑 **Credentials loaded.** Ready to authenticate through your Google Account:")
                
                # Render the Google Sign-in Button with target="_blank" to open in a new tab
                button_html = f"""
                <div style="display: flex; justify-content: center; margin: 20px 0;">
                    <a href="{google_auth_url}" target="_blank" style="text-decoration: none; width: 100%;">
                        <div style="display: flex; align-items: center; justify-content: center; background-color: white; color: #3c4043; border-radius: 4px; padding: 12px; font-weight: 500; font-family: Roboto, sans-serif; cursor: pointer; border: 1px solid #dadce0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: background-color 0.2s;">
                            <img src="https://www.vectorlogo.zone/logos/google/google-icon.svg" width="20" height="20" style="margin-right: 12px;"/>
                            <span style="font-size: 0.95rem; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; color: #374151;">Sign in with Google</span>
                        </div>
                    </a>
                </div>
                """
                st.markdown(button_html, unsafe_allow_html=True)
                
                # Show an option to reset/change credentials
                with st.expander("🔧 Update/Change Google Credentials"):
                    new_client_id = st.text_input("Google Client ID", value=client_id, key="update_client_id")
                    new_client_secret = st.text_input("Google Client Secret", value=client_secret, type="password", key="update_client_secret")
                    new_redirect_uri = st.text_input("Authorized Redirect URI", value=redirect_uri, key="update_redirect_uri")
                    if st.button("💾 Update Credentials", key="btn_update_creds", use_container_width=True):
                        save_google_credentials(new_client_id, new_client_secret, new_redirect_uri)
                        st.success("✅ Credentials updated!")
                        time.sleep(0.5)
                        st.rerun()
                        
            st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'/>", unsafe_allow_html=True)
            
            # Keep a simulated quick login option for testing/development
            with st.expander("🧪 Developer Mode (Quick Local Access)"):
                st.write("Don't have Google credentials? Use these simulated sandbox accounts:")
                
                # Gowtham simulated login
                if st.button("👤 Gowtham (gowtham@saasmetrics.io)", key="btn_g1", use_container_width=True):
                    with st.spinner("Connecting to Google OAuth 2.0 API..."):
                        time.sleep(0.5)
                    # Simulated sign-in path
                    if is_backend_alive:
                        payload = {"name": "Gowtham", "email": "gowtham@saasmetrics.io"}
                        res = requests.post(f"{API_BASE_URL}/auth/google", json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.logged_in = True
                            st.session_state.jwt_token = data["access_token"]
                            st.session_state.user_name = data["user"]["name"]
                            st.session_state.user_email = data["user"]["email"]
                            st.session_state.show_google_modal = False
                            st.session_state.nav_index = 2
                            st.success("✅ Google Sign-in successful!")
                            st.rerun()
                        else:
                            handle_api_error(res)
                    else:
                        conn = get_local_db_conn()
                        cursor = conn.cursor()
                        cursor.execute("SELECT * FROM users WHERE email=?", ("gowtham@saasmetrics.io",))
                        db_user = cursor.fetchone()
                        if not db_user:
                            cursor.execute(
                                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                                ("Gowtham", "gowtham@saasmetrics.io", hash_password("google-oauth-simulated-password"))
                            )
                            conn.commit()
                            cursor.execute("SELECT * FROM users WHERE email=?", ("gowtham@saasmetrics.io",))
                            db_user = cursor.fetchone()
                        conn.close()
                        
                        st.session_state.logged_in = True
                        st.session_state.jwt_token = generate_jwt_token(db_user[0], db_user[1], db_user[2])
                        st.session_state.user_name = db_user[1]
                        st.session_state.user_email = db_user[2]
                        st.session_state.show_google_modal = False
                        st.session_state.nav_index = 2
                        st.success("✅ Google Sign-in successful!")
                        st.rerun()
                
                # Guest Executive simulated login
                if st.button("👤 Guest Executive (executive@saas-corp.com)", key="btn_g2", use_container_width=True):
                    with st.spinner("Connecting to Google OAuth 2.0 API..."):
                        time.sleep(0.5)
                    if is_backend_alive:
                        payload = {"name": "Guest Executive", "email": "executive@saas-corp.com"}
                        res = requests.post(f"{API_BASE_URL}/auth/google", json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.logged_in = True
                            st.session_state.jwt_token = data["access_token"]
                            st.session_state.user_name = data["user"]["name"]
                            st.session_state.user_email = data["user"]["email"]
                            st.session_state.show_google_modal = False
                            st.session_state.nav_index = 2
                            st.success("✅ Google Sign-in successful!")
                            st.rerun()
                        else:
                            handle_api_error(res)
                    else:
                        conn = get_local_db_conn()
                        cursor = conn.cursor()
                        cursor.execute("SELECT * FROM users WHERE email=?", ("executive@saas-corp.com",))
                        db_user = cursor.fetchone()
                        if not db_user:
                            cursor.execute(
                                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                                ("Guest Executive", "executive@saas-corp.com", hash_password("google-oauth-simulated-password"))
                            )
                            conn.commit()
                            cursor.execute("SELECT * FROM users WHERE email=?", ("executive@saas-corp.com",))
                            db_user = cursor.fetchone()
                        conn.close()
                        
                        st.session_state.logged_in = True
                        st.session_state.jwt_token = generate_jwt_token(db_user[0], db_user[1], db_user[2])
                        st.session_state.user_name = db_user[1]
                        st.session_state.user_email = db_user[2]
                        st.session_state.show_google_modal = False
                        st.session_state.nav_index = 2
                        st.success("✅ Google Sign-in successful!")
                        st.rerun()
            
            if st.button("❌ Cancel Google Sign-In", key="btn_g_cancel", use_container_width=True):
                st.session_state.show_google_modal = False
                st.rerun()
    st.stop()

# =========================================================
# HEADER & SIDEBAR NAVIGATION
# =========================================================

# Header
col_logo, col_title = st.columns([1, 15])
with col_logo:
    st.markdown('<div style="margin-top: 10px;"><img src="https://www.vectorlogo.zone/logos/google/google-icon.svg" width="42"/></div>', unsafe_allow_html=True)
with col_title:
    st.title("📊 SaaS Business Analytics & Prediction Platform")
    if st.session_state.logged_in:
        st.markdown(f"💼 *Logged in as:* **{st.session_state.user_name}** ({st.session_state.user_email})")

# Navigation Menu
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h3 style="color: #6366F1; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 700; margin: 0;">SaaS METRICS PRO</h3>
    <span style="color: #00E5FF; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;">PREDICTION ENGINE</span>
</div>
<hr style="border-color: rgba(255,255,255,0.05); margin-top: 10px; margin-bottom: 20px;" />
""", unsafe_allow_html=True)

if not st.session_state.logged_in:
    page = "Login / Signup"
else:
    # Safely retrieve and consume programmatic redirect index after Google Login
    default_idx = 0
    if "nav_index" in st.session_state:
        default_idx = st.session_state.nav_index
        del st.session_state.nav_index # Consume so that subsequent manual clicks operate normally
        
    page = st.sidebar.radio(
        "📊 Analytical Portal",
        [
            "Home", 
            "Upload / Manage Dataset", 
            "Executive SaaS Dashboard", 
            "Cohort Retention Heatmap", 
            "ML Churn Prediction Center"
        ],
        index=default_idx
    )

# Logout Section
if st.session_state.logged_in:
    st.sidebar.markdown("<br><br><hr style='border-color: rgba(255,255,255,0.05);'/>", unsafe_allow_html=True)
    if st.sidebar.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.session_state.jwt_token = ""
        st.session_state.df_details = None
        st.session_state.mapped_columns = {}
        st.session_state.model_trained = False
        st.session_state.model_metrics = {}
        st.session_state.feature_importances = []
        st.session_state.local_df = None
        st.success("✅ Safely Logged Out!")
        st.rerun()

# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("""
    <h2 style="color: white; margin-top: 0;">💡 Empowering Executive Decisions with Predictability</h2>
    <p style="font-size: 1.05rem; line-height: 1.6; color: #CBD5E1;">
        Welcome to the next-generation <strong>SaaS Business Analytics & Churn Prediction Platform</strong>. 
        Engineered specifically for modern subscription-based businesses, this toolkit lets you translate 
        customer attributes and billing metrics into structural cash-flow expansion and churn mitigation workflows.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 15px;">
            <div class="metric-value">MRR</div>
            <div style="color: #94A3B8; font-size: 0.85rem;">Monthly Income Tracking</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 15px;">
            <div class="metric-value">LTV:CAC</div>
            <div style="color: #94A3B8; font-size: 0.85rem;">Unit Economics Health</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 15px;">
            <div class="metric-value">COHORTS</div>
            <div style="color: #94A3B8; font-size: 0.85rem;">Retention Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 15px;">
            <div class="metric-value">AI ML</div>
            <div style="color: #94A3B8; font-size: 0.85rem;">Predictive Risk Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="color: white; margin-top: 20px;">⚡ Key Value Proposition:</h3>
    <ul style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.8;">
        <li><strong>Hybrid Architecture:</strong> Offloads work to a FastAPI service, or automatically operates self-contained inside the cloud browser.</li>
        <li><strong>Multi-Dimensional Cohort Visualizer:</strong> Track user retention vectors across months automatically to pinpoint attrition events.</li>
        <li><strong>Dynamic Column Mapping Engine:</strong> Drop in any standard customer Excel sheet or CSV. Map your parameters easily and see results instantly.</li>
        <li><strong>Random Forest Churn Engine:</strong> Train ML models directly on customer behavior markers (usage days, support tickets, days since last login) to flag at-risk accounts before they cancel.</li>
        <li><strong>Executive Action Simulations:</strong> Simulate LTV adjustments and churn risk scores dynamically using sliders to design tailored discount models or CSM outreach guidelines.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.markdown("<p style='color: #6366F1; font-weight: 600; margin-top: 20px;'>⚠️ Authentication required. Please sign in via the Navigation sidebar.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# LOGIN / SIGNUP PAGE
# =========================================================
elif page == "Login / Signup":
    st.markdown('<div class="premium-card" style="max-width: 550px; margin: 30px auto; padding: 35px !important;">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white; margin-top:0;'>🔐 Security Gateway</h2>", unsafe_allow_html=True)
    
    tab_signin, tab_signup = st.tabs(["Sign In", "Create Local Account"])
    
    with tab_signin:
        email = st.text_input("Corporate Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        
        if st.button("🚪 Sign In", key="btn_signin"):
            if email and password:
                if is_backend_alive:
                    payload = {"email": email, "password": password}
                    res = requests.post(f"{API_BASE_URL}/auth/login", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.logged_in = True
                        st.session_state.jwt_token = data["access_token"]
                        st.session_state.user_name = data["user"]["name"]
                        st.session_state.user_email = data["user"]["email"]
                        st.success("✅ Signed in successfully!")
                        st.rerun()
                    else:
                        handle_api_error(res)
                else:
                    # Local fallback implementation
                    conn = get_local_db_conn()
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
                    db_user = cursor.fetchone()
                    conn.close()
                    
                    if not db_user or not verify_password(password, db_user[3]):
                        st.error("❌ Invalid Email or Password Combination.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.jwt_token = generate_jwt_token(db_user[0], db_user[1], db_user[2])
                        st.session_state.user_name = db_user[1]
                        st.session_state.user_email = db_user[2]
                        st.success("✅ Signed in successfully!")
                        st.rerun()
            else:
                st.warning("⚠️ Please fill in all credentials")
                
        st.markdown("""
        <div style="text-align: center; margin: 20px 0; color: #64748B;">— OR —</div>
        """, unsafe_allow_html=True)
        
        # Google OAuth trigger button
        if st.button("🌐 Sign in with Google", key="btn_google_signin"):
            st.session_state.show_google_modal = True
            st.rerun()
            
    with tab_signup:
        new_name = st.text_input("Full Name", key="reg_name")
        new_email = st.text_input("Corporate Email Address", key="reg_email")
        new_password = st.text_input("Choose Secure Password", type="password", key="reg_pw")
        confirm_pw = st.text_input("Confirm Password", type="password", key="reg_cpw")
        
        if st.button("📝 Create Account", key="btn_signup"):
            if new_name and new_email and new_password and confirm_pw:
                if new_password == confirm_pw:
                    if is_backend_alive:
                        payload = {"name": new_name, "email": new_email, "password": new_password}
                        res = requests.post(f"{API_BASE_URL}/auth/register", json=payload)
                        if res.status_code == 200:
                            st.success("✅ Account created! Please log in under the 'Sign In' tab.")
                        else:
                            handle_api_error(res)
                    else:
                        # Local fallback implementation
                        conn = get_local_db_conn()
                        cursor = conn.cursor()
                        cursor.execute("SELECT * FROM users WHERE email=?", (new_email,))
                        if cursor.fetchone():
                            st.error("❌ An account is already registered with this email.")
                        else:
                            hashed = hash_password(new_password)
                            cursor.execute(
                                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                                (new_name, new_email, hashed)
                            )
                            conn.commit()
                            st.success("✅ Account created! Please log in under the 'Sign In' tab.")
                        conn.close()
                else:
                    st.error("❌ Passwords do not match.")
            else:
                st.warning("⚠️ Please fill in all required fields.")
                
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# UPLOAD / MANAGE DATASET PAGE
# =========================================================
elif page == "Upload / Manage Dataset":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>📂 Data Ingestion Dashboard</h2>", unsafe_allow_html=True)
    
    st.write("To perform detailed business analytics and churn predictions, upload your current subscription dataset or test with our pre-packaged SaaS mock template.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Upload Excel or CSV File")
        uploaded_file = st.file_uploader(
            "Select CSV or Excel (.xlsx) file", 
            type=["csv", "xlsx"],
            help="File must contain customer records with pricing and usage statistics."
        )
        
        if uploaded_file is not None:
            try:
                if is_backend_alive:
                    # Post file as multipart form-data to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(
                        f"{API_BASE_URL}/data/upload", 
                        files=files, 
                        headers=get_auth_headers()
                    )
                    if res.status_code == 200:
                        st.session_state.df_details = res.json()
                        st.session_state.model_trained = False  # Reset ML model on new upload
                        st.success(f"✅ Ingested '{uploaded_file.name}' — Loaded {st.session_state.df_details['rows']} rows and {len(st.session_state.df_details['columns'])} columns successfully.")
                    else:
                        handle_api_error(res)
                else:
                    # Local fallback implementation
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.local_df = df
                    st.session_state.df_details = {
                        "filename": uploaded_file.name,
                        "rows": df.shape[0],
                        "columns": list(df.columns),
                        "preview_json": df.head(5).to_dict(orient="records")
                    }
                    st.session_state.model_trained = False
                    st.success(f"✅ Ingested '{uploaded_file.name}' locally — Loaded {df.shape[0]} rows successfully.")
            except Exception as e:
                st.error(f"❌ Ingestion Error: {e}")
                
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("### No Dataset Handy? Use Our Pre-packaged Template")
        st.write("Click below to dynamically generate a synthetic SaaS customer dataset containing 1,000 active and churned customer accounts with realistic characteristics.")
        
        if st.button("🚀 Load Sample SaaS Dataset"):
            try:
                if is_backend_alive:
                    with st.spinner("Generating beautiful synthetic SaaS database..."):
                        res = requests.post(f"{API_BASE_URL}/data/sample", headers=get_auth_headers())
                    if res.status_code == 200:
                        st.session_state.df_details = res.json()
                        st.session_state.model_trained = False
                        # Auto map sample columns
                        st.session_state.mapped_columns = {
                            "CustomerID": "CustomerID",
                            "SignupDate": "SignupDate",
                            "PlanType": "PlanType",
                            "MonthlyRevenue": "MonthlyRevenue",
                            "Status": "Status",
                            "TenureMonths": "TenureMonths",
                            "UsageFrequency": "UsageFrequency",
                            "SupportTickets": "SupportTickets",
                            "LastLoginDaysAgo": "LastLoginDaysAgo",
                            "LTV": "LTV",
                            "CAC": "CAC"
                        }
                        st.success("✅ Sample SaaS Dataset Loaded and Auto-Mapped successfully!")
                    else:
                        handle_api_error(res)
                else:
                    # Local fallback implementation
                    with st.spinner("Generating beautiful synthetic SaaS database..."):
                        df = generate_sample_saas_data()
                    st.session_state.local_df = df
                    st.session_state.df_details = {
                        "filename": "sample_saas_dataset.csv",
                        "rows": df.shape[0],
                        "columns": list(df.columns),
                        "preview_json": df.head(5).to_dict(orient="records")
                    }
                    st.session_state.model_trained = False
                    st.session_state.mapped_columns = {
                        "CustomerID": "CustomerID",
                        "SignupDate": "SignupDate",
                        "PlanType": "PlanType",
                        "MonthlyRevenue": "MonthlyRevenue",
                        "Status": "Status",
                        "TenureMonths": "TenureMonths",
                        "UsageFrequency": "UsageFrequency",
                        "SupportTickets": "SupportTickets",
                        "LastLoginDaysAgo": "LastLoginDaysAgo",
                        "LTV": "LTV",
                        "CAC": "CAC"
                    }
                    st.success("✅ Sample SaaS Dataset Loaded locally and Auto-Mapped successfully!")
            except Exception as e:
                st.error(f"❌ Generation Error: {e}")
            
    with col2:
        st.markdown("### Interactive Column Mapper")
        st.write("Map your file columns to standard SaaS variables to ensure accurate dashboard visualizer calculations.")
        
        if st.session_state.df_details is None:
            st.info("⚠️ Please upload a dataset or load the sample dataset to map columns.")
        else:
            df_cols = list(st.session_state.df_details["columns"])
            
            # Mapping Interface
            st.session_state.mapped_columns["CustomerID"] = st.selectbox(
                "Customer Identifier Column", 
                df_cols, 
                index=df_cols.index("CustomerID") if "CustomerID" in df_cols else 0
            )
            st.session_state.mapped_columns["SignupDate"] = st.selectbox(
                "Signup / Subscription Start Date", 
                df_cols, 
                index=df_cols.index("SignupDate") if "SignupDate" in df_cols else 0
            )
            st.session_state.mapped_columns["PlanType"] = st.selectbox(
                "Subscription Plan Tier", 
                df_cols, 
                index=df_cols.index("PlanType") if "PlanType" in df_cols else 0
            )
            st.session_state.mapped_columns["MonthlyRevenue"] = st.selectbox(
                "Monthly Revenue / Price Column ($)", 
                df_cols, 
                index=df_cols.index("MonthlyRevenue") if "MonthlyRevenue" in df_cols else 0
            )
            st.session_state.mapped_columns["Status"] = st.selectbox(
                "Account Status (Active vs. Churned)", 
                df_cols, 
                index=df_cols.index("Status") if "Status" in df_cols else 0
            )
            st.session_state.mapped_columns["TenureMonths"] = st.selectbox(
                "Tenure (Months Subscribed)", 
                df_cols, 
                index=df_cols.index("TenureMonths") if "TenureMonths" in df_cols else 0
            )
            st.session_state.mapped_columns["SupportTickets"] = st.selectbox(
                "Support Tickets Raised (ML Feature)", 
                df_cols, 
                index=df_cols.index("SupportTickets") if "SupportTickets" in df_cols else 0
            )
            st.session_state.mapped_columns["UsageFrequency"] = st.selectbox(
                "Active Login Days / Month (ML Feature)", 
                df_cols, 
                index=df_cols.index("UsageFrequency") if "UsageFrequency" in df_cols else 0
            )
            st.session_state.mapped_columns["LastLoginDaysAgo"] = st.selectbox(
                "Days Since Last Login (ML Feature)", 
                df_cols, 
                index=df_cols.index("LastLoginDaysAgo") if "LastLoginDaysAgo" in df_cols else 0
            )
            st.session_state.mapped_columns["LTV"] = st.selectbox(
                "Customer Lifetime Value (LTV)", 
                df_cols, 
                index=df_cols.index("LTV") if "LTV" in df_cols else 0
            )
            st.session_state.mapped_columns["CAC"] = st.selectbox(
                "Customer Acquisition Cost (CAC)", 
                df_cols, 
                index=df_cols.index("CAC") if "CAC" in df_cols else 0
            )
            
            st.markdown("""
            <div style="background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); padding: 12px; border-radius: 8px; margin-top: 15px;">
                <p style="margin: 0; font-size: 0.85rem; color: #00E5FF; font-weight: 600;">💡 Columns are successfully mapped and saved to session memory!</p>
            </div>
            """, unsafe_allow_html=True)
            
    # File details & preview
    if st.session_state.df_details is not None:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'/>", unsafe_allow_html=True)
        st.markdown("### Ingested Dataset Preview (First 5 Rows)")
        preview_df = pd.DataFrame(st.session_state.df_details["preview_json"])
        st.dataframe(preview_df, use_container_width=True)
        
        # In-memory cleaner for Local Fallback Mode
        if not is_backend_alive:
            st.subheader("🛠️ Clean Dataset in Memory")
            st.write("Ensure there are no blank cells. Check below to perform zero-mean numeric imputation for blank entries.")
            if st.button("Apply In-Place Cleaning"):
                st.session_state.local_df = st.session_state.local_df.fillna(st.session_state.local_df.mean(numeric_only=True))
                st.session_state.df_details["preview_json"] = st.session_state.local_df.head(5).to_dict(orient="records")
                st.success("✅ Local dataset blanks filled successfully!")
                st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXECUTIVE SAAS DASHBOARD PAGE
# =========================================================
elif page == "Executive SaaS Dashboard":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>📈 Executive SaaS Financials & Health KPI Dashboard</h2>", unsafe_allow_html=True)
    
    if st.session_state.df_details is None:
        st.warning("⚠️ Upload a SaaS dataset or load our template under 'Upload / Manage Dataset' before viewing analytics.")
    else:
        try:
            m = st.session_state.mapped_columns
            
            # 1. Fetch KPIs
            if is_backend_alive:
                res_kpis = requests.post(
                    f"{API_BASE_URL}/analytics/kpis", 
                    json=m,
                    headers=get_auth_headers()
                )
                if res_kpis.status_code != 200:
                    handle_api_error(res_kpis)
                    st.stop()
                kpis = res_kpis.json()
            else:
                # Local fallback implementation
                df_clean = preprocess_dataframe(st.session_state.local_df, m)
                kpis = compute_saas_kpis(df_clean, m)
            
            # Display KPI Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">${kpis["latest_mrr"]:,.2f}</div>
                    <div class="metric-label">Monthly Rec. Revenue (MRR)</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">${kpis["latest_arr"]:,.2f}</div>
                    <div class="metric-label">Annual Rec. Revenue (ARR)</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">${kpis["arpu"]:,.2f}</div>
                    <div class="metric-label">Avg Rev Per User (ARPU)</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">{kpis["active_count"]:,} / {kpis["total_customers"]:,}</div>
                    <div class="metric-label">Active Users Ratio</div>
                </div>
                """, unsafe_allow_html=True)
                
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">{kpis["logo_churn_rate"]:.2f}%</div>
                    <div class="metric-label">Customer Logo Churn</div>
                </div>
                """, unsafe_allow_html=True)
            with c6:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">{kpis["revenue_churn_rate"]:.2f}%</div>
                    <div class="metric-label">Gross Revenue Churn</div>
                </div>
                """, unsafe_allow_html=True)
            with c7:
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value">${kpis["avg_ltv"]:,.2f} / ${kpis["avg_cac"]:,.2f}</div>
                    <div class="metric-label">LTV / CAC Average</div>
                </div>
                """, unsafe_allow_html=True)
            with c8:
                # Color code LTV:CAC
                ratio = kpis["ltv_cac_ratio"]
                ratio_color = "#10B981" if ratio >= 3.0 else ("#F59E0B" if ratio >= 1.0 else "#EF4444")
                st.markdown(f"""
                <div class="premium-card" style="text-align: center; margin-bottom: 15px; padding: 15px !important;">
                    <div class="metric-value" style="background: none; -webkit-text-fill-color: {ratio_color}; color: {ratio_color};">{ratio:.2f}x</div>
                    <div class="metric-label">LTV to CAC Ratio</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 25px 0;'/>", unsafe_allow_html=True)
            
            # 2. Fetch Charts Data
            if is_backend_alive:
                res_charts = requests.post(
                    f"{API_BASE_URL}/analytics/charts", 
                    json=m,
                    headers=get_auth_headers()
                )
                if res_charts.status_code != 200:
                    handle_api_error(res_charts)
                    st.stop()
                charts = res_charts.json()
            else:
                # Local fallback implementation
                df_clean = preprocess_dataframe(st.session_state.local_df, m)
                charts = get_charts_data(df_clean, m)
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 Plan Type & Revenue Distribution")
                plan_dist = charts["plan_distribution"]
                
                # Plotly Donut Chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=plan_dist["labels"], 
                    values=plan_dist["revenue"], 
                    hole=.4,
                    marker=dict(colors=["#6366F1", "#00E5FF", "#EC4899"])
                )])
                fig_donut.update_layout(
                    title="Revenue Contribution by Subscription Plan ($)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#F3F4F6",
                    showlegend=True,
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with col_chart2:
                st.subheader("📅 Customer Inflow Growth Trend")
                signup_trend = charts["signup_trend"]
                
                # Plotly Bar Chart
                fig_growth = px.bar(
                    x=signup_trend["months"], 
                    y=signup_trend["acquisitions"],
                    title="Monthly Customer Acquisitions",
                    color_discrete_sequence=["#00E5FF"]
                )
                fig_growth.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#F3F4F6",
                    xaxis=dict(showgrid=False, title="Acquisitions Month"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Acquisitions"),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_growth, use_container_width=True)
                
            col_chart3, col_chart4 = st.columns(2)
            with col_chart3:
                st.subheader("🎟️ Support Load vs. Churn Rate Correlation")
                support_corr = charts["support_correlation"]
                
                fig_support = px.line(
                    x=support_corr["tickets"], 
                    y=support_corr["churn_rate"],
                    title="Churn Probability by Support Tickets Raised",
                    markers=True,
                    color_discrete_sequence=["#EF4444"]
                )
                fig_support.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#F3F4F6",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Support Tickets Raised"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Churn Rate (%)"),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_support, use_container_width=True)
                
            with col_chart4:
                st.subheader("💡 Smart Executive Insights & Actions")
                
                insights = []
                if kpis["ltv_cac_ratio"] < 3.0:
                    insights.append("⚠️ **Critical Unit Economics:** Your LTV to CAC ratio is currently below the industry benchmark of 3.0x. Consider reducing marketing spends or increasing monthly prices.")
                else:
                    insights.append("🚀 **Excellent Unit Economics:** Your LTV:CAC is highly positive. Expand search ads immediately to capture higher market shares.")
                    
                # Support Tickets evaluation
                tickets_arr = support_corr["churn_rate"]
                if len(tickets_arr) > 4 and tickets_arr[4] > 50:
                    insights.append("🔍 **Customer Experience Friction:** Churn rates spike above 50% for users with more than 4 support tickets. Implement a proactive outreach program for high-ticket clients.")
                
                for insight in insights:
                    st.info(insight)
                    
        except Exception as e:
            st.error(f"❌ Analysis Error: Verify your column mapping settings. Detail: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# COHORT RETENTION HEATMAP PAGE
# =========================================================
elif page == "Cohort Retention Heatmap":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>📅 Cohort Retention Decay Matrix</h2>", unsafe_allow_html=True)
    
    if st.session_state.df_details is None:
        st.warning("⚠️ Upload a SaaS dataset or load our template under 'Upload / Manage Dataset' before viewing cohort analysis.")
    else:
        try:
            m = st.session_state.mapped_columns
            
            # Fetch cohorts matrix
            if is_backend_alive:
                res_cohorts = requests.post(
                    f"{API_BASE_URL}/analytics/cohorts", 
                    json=m,
                    headers=get_auth_headers()
                )
                if res_cohorts.status_code != 200:
                    handle_api_error(res_cohorts)
                    st.stop()
                cohorts = res_cohorts.json()
            else:
                # Local fallback implementation
                df_clean = preprocess_dataframe(st.session_state.local_df, m)
                cohorts = get_cohort_matrix_data(df_clean, m)
            
            # Plot Cohort Heatmap
            fig_heatmap = ff.create_annotated_heatmap(
                z=cohorts["matrix"],
                x=cohorts["months"],
                y=cohorts["cohorts"],
                colorscale="Viridis",
                showscale=True
            )
            fig_heatmap.update_layout(
                title="Customer Cohort Retention Decay (%)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F3F4F6",
                margin=dict(t=50, b=10, l=10, r=10),
                height=500
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("""
            ### 📘 How to Read Cohort Decay Matrices
            <p style='color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;'>
                Each row represents a **Cohort** of users who signed up in the same calendar month. 
                Moving left-to-right displays the percentage of those original users who remained active as months progressed.
                A vertical color drop-off (e.g. sharp declines from Month 0 to Month 1) suggests immediate onboarding issues, 
                while gradual fades indicate standard usage cycles.
            </p>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Cohort Error: Verify Date fields and tenure mapping. Detail: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ML CHURN PREDICTION CENTER PAGE
# =========================================================
elif page == "ML Churn Prediction Center":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>🤖 Machine Learning Churn Predictor & Simulator</h2>", unsafe_allow_html=True)
    
    if st.session_state.df_details is None:
        st.warning("⚠️ Upload a SaaS dataset or load our template under 'Upload / Manage Dataset' before training the predictor.")
    else:
        # ML controls in columns
        col_ctrl, col_metrics = st.columns([1, 1])
        
        with col_ctrl:
            st.subheader("⚙️ Model Trainer")
            st.write("Click below to train a Random Forest Classifier on historical subscription details and usage habits.")
            
            m = st.session_state.mapped_columns
            ml_features = [m["TenureMonths"], m["UsageFrequency"], m["SupportTickets"], m["LastLoginDaysAgo"], m["PlanType"], m["MonthlyRevenue"]]
            st.write(f"**Model Features:** *{', '.join(ml_features)}*")
            st.write(f"**Target Column:** *{m['Status']}*")
            
            if st.button("🏋️ Train Random Forest Model"):
                with st.spinner("Executing Feature Engineering and Model Training..."):
                    try:
                        if is_backend_alive:
                            res = requests.post(
                                f"{API_BASE_URL}/ml/train", 
                                json=m,
                                headers=get_auth_headers()
                            )
                            if res.status_code == 200:
                                data = res.json()
                                st.session_state.model_trained = True
                                st.session_state.model_metrics = data["metrics"]
                                st.session_state.feature_importances = data["feature_importances"]
                                st.success("✅ Random Forest Model trained successfully on backend!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                handle_api_error(res)
                        else:
                            # Local fallback implementation
                            df_clean = preprocess_dataframe(st.session_state.local_df, m)
                            data = train_churn_model(df_clean, m)
                            st.session_state.model_trained = True
                            st.session_state.model_metrics = data["metrics"]
                            st.session_state.feature_importances = data["feature_importances"]
                            st.success("✅ Random Forest Model trained successfully in-app!")
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
                        
        with col_metrics:
            st.subheader("📈 Performance Diagnostics")
            if not st.session_state.model_trained:
                st.info("🕒 Train the classifier model to calculate precision, recall, and metrics.")
            else:
                metrics = st.session_state.model_metrics
                
                # Show cards
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
                    <div class="premium-card" style="text-align: center; margin-bottom: 10px; padding: 12px !important; border-color: rgba(0, 229, 255, 0.3) !important;">
                        <div class="metric-value" style="font-size:1.8rem;">{metrics['Accuracy']:.1f}%</div>
                        <div class="metric-label" style="font-size:0.75rem;">Model Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="premium-card" style="text-align: center; margin-bottom: 10px; padding: 12px !important; border-color: rgba(99, 102, 241, 0.3) !important;">
                        <div class="metric-value" style="font-size:1.8rem;">{metrics['Precision']:.1f}%</div>
                        <div class="metric-label" style="font-size:0.75rem;">Model Precision</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                col_m3, col_m4 = st.columns(2)
                with col_m3:
                    st.markdown(f"""
                    <div class="premium-card" style="text-align: center; margin-bottom: 10px; padding: 12px !important;">
                        <div class="metric-value" style="font-size:1.8rem;">{metrics['Recall']:.1f}%</div>
                        <div class="metric-label" style="font-size:0.75rem;">Model Recall</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f"""
                    <div class="premium-card" style="text-align: center; margin-bottom: 10px; padding: 12px !important;">
                        <div class="metric-value" style="font-size:1.8rem;">{metrics['F1-Score']:.1f}%</div>
                        <div class="metric-label" style="font-size:0.75rem;">F1-Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Feature Importance chart
        if st.session_state.model_trained:
            st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 25px 0;'/>", unsafe_allow_html=True)
            st.subheader("🔥 Key Churn Indicators (Feature Importance)")
            
            feat_imp = st.session_state.feature_importances
            fig_imp = px.bar(
                x=[f["Importance"] for f in feat_imp],
                y=[f["Feature"] for f in feat_imp],
                orientation="h",
                color=[f["Importance"] for f in feat_imp],
                color_continuous_scale="Purples",
                title="Drivers of Churn as Identified by ML Algorithm"
            )
            fig_imp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F3F4F6",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Importance Weight"),
                yaxis=dict(showgrid=False, title="Feature"),
                margin=dict(t=40, b=10, l=10, r=10),
                height=250
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Active Customer risk Ledger & Simulator Tabs
            st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 25px 0;'/>", unsafe_allow_html=True)
            
            tab_ledger, tab_simulator = st.tabs(["📋 Active Customer Churn Ledger", "🔮 Sandbox Customer Churn Simulator"])
            
            with tab_ledger:
                st.subheader("Active Customer Risk Tracking Ledger")
                st.write("Below is a ranked list of current active customers sorted by predicted churn probability. Outreached accounts can be targeted by CSMs immediately.")
                
                try:
                    if is_backend_alive:
                        res_ledger = requests.post(
                            f"{API_BASE_URL}/ml/ledger", 
                            json=m,
                            headers=get_auth_headers()
                        )
                        if res_ledger.status_code == 200:
                            ledger_data = res_ledger.json()
                        else:
                            handle_api_error(res_ledger)
                            st.stop()
                    else:
                        # Local fallback implementation
                        df_clean = preprocess_dataframe(st.session_state.local_df, m)
                        ledger_data = get_active_ledger_predictions(df_clean, m)
                    
                    if not ledger_data:
                        st.info("No active customer records available in the dataset.")
                    else:
                        # Parse into dataframe
                        display_ledger = pd.DataFrame(ledger_data)
                        
                        # Clean ChurnRiskProbability format
                        display_ledger["ChurnRiskProbability"] = display_ledger["ChurnRiskProbability"].apply(lambda p: f"{p*100:.1f}%")
                        
                        # Search filter
                        search_term = st.text_input("🔍 Search Active Customer Ledger by ID", "")
                        if search_term:
                            display_ledger = display_ledger[display_ledger["CustomerID"].astype(str).str.contains(search_term, case=False)]
                            
                        # Show columns
                        st.dataframe(
                            display_ledger[[
                                "CustomerID", "PlanType", "MonthlyRevenue", "TenureMonths", 
                                "SupportTickets", "LastLoginDaysAgo", "ChurnRiskProbability", "RiskCategory"
                            ]], 
                            use_container_width=True
                        )
                except Exception as ex:
                    st.error(f"Failed to generate risk ledger. Detail: {ex}")
                    
            with tab_simulator:
                st.subheader("Customer Attrition Sandbox Simulator")
                st.write("Toggle parameters below to predict the churn risk probability of an active customer account dynamically.")
                
                col_sim1, col_sim2 = st.columns([1, 1])
                
                with col_sim1:
                    sim_plan = st.selectbox("Plan Type Tier", ["Starter", "Growth", "Enterprise"])
                    sim_revenue = st.slider("Monthly Revenue ($)", 10.0, 500.0, value=99.0)
                    sim_tenure = st.slider("Subscription Tenure (Months)", 1, 36, value=6)
                    sim_usage = st.slider("Active Logins (Days per Month)", 0, 30, value=25)
                    sim_tickets = st.slider("Support Tickets Raised", 0, 15, value=2)
                    sim_last_login = st.slider("Days Since Last Login", 0, 90, value=3)
                    
                with col_sim2:
                    st.markdown("<div style='text-align:center; padding-top:20px;'>", unsafe_allow_html=True)
                    st.write("### SIMULATED RISK REPORT")
                    
                    if st.button("🔮 Calculate Simulated Churn Risk"):
                        try:
                            if is_backend_alive:
                                # Call sandbox inference REST endpoint on backend
                                payload = {
                                    "plan": sim_plan,
                                    "revenue": sim_revenue,
                                    "tenure": sim_tenure,
                                    "usage": sim_usage,
                                    "tickets": sim_tickets,
                                    "last_login": sim_last_login
                                }
                                res_sim = requests.post(
                                    f"{API_BASE_URL}/ml/simulate", 
                                    json=payload,
                                    headers=get_auth_headers()
                                )
                                if res_sim.status_code == 200:
                                    sim_data = res_sim.json()
                                    prob = sim_data["probability"]
                                    risk_color = sim_data["risk_color"]
                                    risk_text = sim_data["risk_text"]
                                    protocol = sim_data["protocol"]
                                else:
                                    handle_api_error(res_sim)
                                    st.stop()
                            else:
                                # Local fallback implementation
                                sim_data = simulate_customer_churn_risk(
                                    sim_plan=sim_plan,
                                    sim_revenue=sim_revenue,
                                    sim_tenure=sim_tenure,
                                    sim_usage=sim_usage,
                                    sim_tickets=sim_tickets,
                                    sim_last_login=sim_last_login
                                )
                                prob = sim_data["probability"]
                                risk_color = sim_data["risk_color"]
                                risk_text = sim_data["risk_text"]
                                protocol = sim_data["protocol"]
                            
                            st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.8); border: 2px solid {risk_color}; border-radius: 12px; padding: 25px; margin-top:15px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                                <div style="font-size: 1.1rem; color: #94A3B8; text-transform:uppercase;">Predicted Churn Probability</div>
                                <div style="font-size: 3.5rem; font-weight: 800; color: {risk_color}; margin: 10px 0;">{prob*100:.1f}%</div>
                                <div style="display: inline-block; background-color: {risk_color}22; border: 1px solid {risk_color}; color: {risk_color}; font-weight: 700; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem;">
                                    {risk_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Render suggested CSM actions
                            st.markdown("#### Suggested Retention Protocol:")
                            if prob >= 0.7:
                                st.error(protocol)
                            elif prob >= 0.4:
                                st.warning(protocol)
                            else:
                                st.success(protocol)
                                
                        except Exception as ex_sim:
                            st.error(f"Simulation execution error: {ex_sim}")
                            
                    st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
import time

from backend.database import get_db_connection
from backend.auth import hash_password, verify_password, generate_jwt_token, decode_jwt_token
from backend.upload import save_uploaded_file, read_file_to_df, generate_sample_saas_data, _dataframe_cache
from backend.analytics import preprocess_dataframe, compute_saas_kpis, get_charts_data, get_cohort_matrix_data
from backend.ml.churn_model import train_churn_model, get_active_ledger_predictions, simulate_customer_churn_risk

app = FastAPI(
    title="SaaS Business Analytics Platform Backend",
    description="REST API service for user authing, CSV uploading, business intelligence, and ML churn prediction",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SCHEMAS
# =========================================================
class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class GoogleLogin(BaseModel):
    name: str
    email: str

class ColumnMapping(BaseModel):
    CustomerID: str
    SignupDate: str
    PlanType: str
    MonthlyRevenue: str
    Status: str
    TenureMonths: str
    SupportTickets: str
    UsageFrequency: str
    LastLoginDaysAgo: str
    LTV: str
    CAC: str

class SimulationInput(BaseModel):
    plan: str
    revenue: float
    tenure: int
    usage: int
    tickets: int
    last_login: int

# =========================================================
# DEPENDENCIES (JWT AUTHENTICATION)
# =========================================================
def get_current_user(authorization: str = Header(None)):
    """Validates the JWT token in Authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    token = authorization.split(" ")[1]
    payload = decode_jwt_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Expired or compromised authentication token.")
    
    return payload

# =========================================================
# AUTH ROUTES
# =========================================================
@app.post("/api/auth/register")
def register_user(user: UserRegister):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email=?", (user.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="An account is already registered with this email.")
        
    hashed = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (user.name, user.email, hashed)
    )
    conn.commit()
    conn.close()
    return {"message": "Account created successfully!"}

@app.post("/api/auth/login")
def login_user(user: UserLogin):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email=?", (user.email,))
    db_user = cursor.fetchone()
    conn.close()
    
    if not db_user or not verify_password(user.password, db_user[3]):
        raise HTTPException(status_code=401, detail="Invalid Email or Password Combination.")
        
    token = generate_jwt_token(db_user[0], db_user[1], db_user[2])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "name": db_user[1],
            "email": db_user[2]
        }
    }

@app.post("/api/auth/google")
def login_with_google(user: GoogleLogin):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if Google email exists, if not, create a placeholder account
    cursor.execute("SELECT * FROM users WHERE email=?", (user.email,))
    db_user = cursor.fetchone()
    
    if not db_user:
        # Generate hash for google auth placeholder password
        google_hash = hash_password("google-oauth-simulated-password")
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (user.name, user.email, google_hash)
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE email=?", (user.email,))
        db_user = cursor.fetchone()
        
    conn.close()
    
    token = generate_jwt_token(db_user[0], db_user[1], db_user[2])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "name": db_user[1],
            "email": db_user[2]
        }
    }

# =========================================================
# DATA & INGESTION ROUTES
# =========================================================
@app.post("/api/data/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        content = await file.read()
        file_path = save_uploaded_file(content, file.filename)
        df = read_file_to_df(file_path)
        
        # Save dataframe to memory cache mapped to user's email
        email = current_user["email"]
        _dataframe_cache[email] = df
        
        return {
            "filename": file.filename,
            "rows": df.shape[0],
            "columns": list(df.columns),
            "preview_json": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest dataset: {str(e)}")

@app.post("/api/data/sample")
def load_sample_dataset(current_user: dict = Depends(get_current_user)):
    try:
        df = generate_sample_saas_data()
        email = current_user["email"]
        _dataframe_cache[email] = df
        return {
            "filename": "sample_saas_dataset.csv",
            "rows": df.shape[0],
            "columns": list(df.columns),
            "preview_json": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample dataset: {str(e)}")

# =========================================================
# ANALYTICS ROUTERS
# =========================================================
def _get_user_df(email: str):
    """Safely retrieves the current user's active cached dataframe"""
    if email not in _dataframe_cache:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a file first.")
    return _dataframe_cache[email]

@app.post("/api/analytics/kpis")
def get_kpi_metrics(mapping: ColumnMapping, current_user: dict = Depends(get_current_user)):
    try:
        raw_df = _get_user_df(current_user["email"])
        m = mapping.dict()
        df = preprocess_dataframe(raw_df, m)
        kpis = compute_saas_kpis(df, m)
        return kpis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI calculations failed: {str(e)}")

@app.post("/api/analytics/charts")
def get_visualizations_data(mapping: ColumnMapping, current_user: dict = Depends(get_current_user)):
    try:
        raw_df = _get_user_df(current_user["email"])
        m = mapping.dict()
        df = preprocess_dataframe(raw_df, m)
        charts_data = get_charts_data(df, m)
        return charts_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visuals data compilation failed: {str(e)}")

@app.post("/api/analytics/cohorts")
def get_cohorts_matrix(mapping: ColumnMapping, current_user: dict = Depends(get_current_user)):
    try:
        raw_df = _get_user_df(current_user["email"])
        m = mapping.dict()
        df = preprocess_dataframe(raw_df, m)
        cohorts_data = get_cohort_matrix_data(df, m)
        return cohorts_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohorts compilation failed: {str(e)}")

# =========================================================
# MACHINE LEARNING ROUTERS
# =========================================================
@app.post("/api/ml/train")
def train_machine_learning_model(mapping: ColumnMapping, current_user: dict = Depends(get_current_user)):
    try:
        raw_df = _get_user_df(current_user["email"])
        m = mapping.dict()
        df = preprocess_dataframe(raw_df, m)
        metrics = train_churn_model(df, m)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML training execution failed: {str(e)}")

@app.post("/api/ml/ledger")
def get_churn_risk_ledger(mapping: ColumnMapping, current_user: dict = Depends(get_current_user)):
    try:
        raw_df = _get_user_df(current_user["email"])
        m = mapping.dict()
        df = preprocess_dataframe(raw_df, m)
        ledger_data = get_active_ledger_predictions(df, m)
        return ledger_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate active risk ledger: {str(e)}")

@app.post("/api/ml/simulate")
def simulate_churn_inference(sim: SimulationInput, current_user: dict = Depends(get_current_user)):
    try:
        inference = simulate_customer_churn_risk(
            sim_plan=sim.plan,
            sim_revenue=sim.revenue,
            sim_tenure=sim.tenure,
            sim_usage=sim.usage,
            sim_tickets=sim.tickets,
            sim_last_login=sim.last_login
        )
        return inference
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sandbox inference failed: {str(e)}")

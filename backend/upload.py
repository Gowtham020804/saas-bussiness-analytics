import pandas as pd
import numpy as np
import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory dataframe cache mapping user_email to active pandas DataFrame
_dataframe_cache = {}

def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """Saves the uploaded file binary to backend/uploads/ folder"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

def read_file_to_df(file_path: str) -> pd.DataFrame:
    """Reads a CSV or Excel file into a pandas DataFrame"""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)

def generate_sample_saas_data() -> pd.DataFrame:
    """Generates high-fidelity synthetic SaaS database matching logical churn patterns"""
    np.random.seed(42)
    n_customers = 1000
    
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_customers)]
    
    # Generate signup dates over the last 24 months
    today = datetime.date(2026, 6, 2)
    signup_dates = []
    for _ in range(n_customers):
        days_ago = np.random.randint(30, 730)
        date = today - datetime.timedelta(days=days_ago)
        signup_dates.append(date)
        
    plans = ["Starter", "Growth", "Enterprise"]
    plan_probs = [0.6, 0.3, 0.1]
    customer_plans = np.random.choice(plans, size=n_customers, p=plan_probs)
    
    plan_prices = {"Starter": 29.0, "Growth": 99.0, "Enterprise": 299.0}
    plan_cac = {"Starter": 90.0, "Growth": 300.0, "Enterprise": 1200.0}
    
    monthly_revenue = [round(plan_prices[p] + np.random.uniform(-5.0, 15.0), 2) for p in customer_plans]
    cac = [round(plan_cac[p] + np.random.uniform(-20.0, 50.0), 2) for p in customer_plans]
    
    # Usage metrics
    usage_frequency = np.random.randint(1, 31, size=n_customers) # active days per month
    support_tickets = np.random.randint(0, 11, size=n_customers)
    last_login_days = np.random.randint(0, 45, size=n_customers)
    
    # Churn Scoring Mechanism (Logically correlated)
    churn_score = (
        (support_tickets * 0.18) 
        + (last_login_days * 0.03) 
        - (usage_frequency * 0.02) 
        + np.random.normal(0, 0.25, size=n_customers)
    )
    
    status = []
    tenure_months = []
    
    for i in range(n_customers):
        months_since_signup = max(1, int((today - signup_dates[i]).days / 30.4))
        
        # Base Churn determination
        if churn_score[i] > 0.5:
            st = "Churned"
            # Churned tenure must be less than signup timeline
            ten = np.random.randint(1, months_since_signup + 1)
        else:
            st = "Active"
            ten = months_since_signup
            
        status.append(st)
        tenure_months.append(ten)
        
    ltv = [round(mr * ten, 2) for mr, ten in zip(monthly_revenue, tenure_months)]
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "SignupDate": [d.strftime("%Y-%m-%d") for d in signup_dates],
        "PlanType": customer_plans,
        "MonthlyRevenue": monthly_revenue,
        "Status": status,
        "TenureMonths": tenure_months,
        "UsageFrequency": usage_frequency,
        "SupportTickets": support_tickets,
        "LastLoginDaysAgo": last_login_days,
        "LTV": ltv,
        "CAC": cac
    })
    
    return df

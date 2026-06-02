import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "churn_rf_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "plan_encoder.joblib")

# Keep active encoders in memory
_in_memory_clf = None
_in_memory_encoder = None

def train_churn_model(df, m):
    """Trains the RandomForestClassifier, saves model binary, and returns evaluation diagnostics"""
    global _in_memory_clf, _in_memory_encoder
    
    ml_df = df.copy()
    ml_features = [m["TenureMonths"], m["UsageFrequency"], m["SupportTickets"], m["LastLoginDaysAgo"], m["PlanType"], m["MonthlyRevenue"]]
    
    # 1. Encode Target Status (Churned/Inactive = 1, Active = 0)
    target_col = m["Status"]
    ml_df["target"] = ml_df[target_col].astype(str).str.lower().apply(
        lambda x: 1 if x in ["churned", "inactive"] else 0
    )
    
    # 2. Encode Plan Type
    le_plan = LabelEncoder()
    ml_df[m["PlanType"]] = le_plan.fit_transform(ml_df[m["PlanType"]].astype(str))
    
    # 3. Features & Target
    X = ml_df[ml_features]
    y = ml_df["target"]
    
    # 4. Train-Test Split (stratify to handle class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    
    # 5. Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Predict & Evaluate
    preds = clf.predict(X_test)
    
    # Metrics
    accuracy = float(accuracy_score(y_test, preds) * 100)
    precision = float(precision_score(y_test, preds, zero_division=0) * 100)
    recall = float(recall_score(y_test, preds, zero_division=0) * 100)
    f1 = float(f1_score(y_test, preds, zero_division=0) * 100)
    
    # Save Model Weights & Encoders
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le_plan, ENCODER_PATH)
    
    _in_memory_clf = clf
    _in_memory_encoder = le_plan
    
    # Feature Importances
    importances = list(clf.feature_importances_)
    feature_importances = [{"Feature": feat, "Importance": float(imp)} for feat, imp in zip(ml_features, importances)]
    # Sort descending
    feature_importances = sorted(feature_importances, key=lambda x: x["Importance"], reverse=True)
    
    return {
        "metrics": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        },
        "feature_importances": feature_importances
    }

def get_active_ledger_predictions(df, m):
    """Predicts churn risk probabilities for all active customers in the database"""
    global _in_memory_clf, _in_memory_encoder
    
    # Load model from disk if in-memory is empty
    if _in_memory_clf is None:
        if os.path.exists(MODEL_PATH):
            _in_memory_clf = joblib.load(MODEL_PATH)
            _in_memory_encoder = joblib.load(ENCODER_PATH)
        else:
            raise ValueError("No model trained yet! Please train the ML model first.")
            
    active_records = df[df[m["Status"]].astype(str).str.lower() == "active"].copy()
    if active_records.empty:
        return []
        
    # Encode Plan
    active_records["plan_encoded"] = _in_memory_encoder.transform(active_records[m["PlanType"]].astype(str))
    
    # Order features matching ml_features: TenureMonths, UsageFrequency, SupportTickets, LastLoginDaysAgo, PlanType, MonthlyRevenue
    ml_features = [m["TenureMonths"], m["UsageFrequency"], m["SupportTickets"], m["LastLoginDaysAgo"], m["PlanType"], m["MonthlyRevenue"]]
    X_active = active_records[[m["TenureMonths"], m["UsageFrequency"], m["SupportTickets"], m["LastLoginDaysAgo"], "plan_encoded", m["MonthlyRevenue"]]].copy()
    X_active.columns = ml_features # align names
    
    # Predict Churn Probabilities
    probs = _in_memory_clf.predict_proba(X_active)
    
    # Single-class safeguard
    if probs.shape[1] > 1:
        churn_probabilities = probs[:, 1]
    else:
        churn_probabilities = [1.0 if _in_memory_clf.classes_[0] == 1 else 0.0] * len(active_records)
        
    active_records["ChurnRiskProbability"] = [float(p) for p in churn_probabilities]
    
    def get_risk_label(p):
        if p >= 0.70: return "🔴 HIGH RISK"
        elif p >= 0.40: return "🟡 MEDIUM RISK"
        return "🟢 SAFE"
        
    active_records["RiskCategory"] = active_records["ChurnRiskProbability"].apply(get_risk_label)
    
    # Sort descending
    active_records = active_records.sort_values("ChurnRiskProbability", ascending=False)
    
    # Select response columns
    response_list = []
    for _, row in active_records.iterrows():
        response_list.append({
            "CustomerID": str(row[m["CustomerID"]]),
            "PlanType": str(row[m["PlanType"]]),
            "MonthlyRevenue": float(row[m["MonthlyRevenue"]]),
            "TenureMonths": int(row[m["TenureMonths"]]),
            "SupportTickets": int(row[m["SupportTickets"]]),
            "LastLoginDaysAgo": int(row[m["LastLoginDaysAgo"]]),
            "ChurnRiskProbability": float(row["ChurnRiskProbability"]),
            "RiskCategory": str(row["RiskCategory"])
        })
        
    return response_list

def simulate_customer_churn_risk(sim_plan, sim_revenue, sim_tenure, sim_usage, sim_tickets, sim_last_login):
    """Performs manual sandbox inference for custom inputs, returning risk gauges and retention recipes"""
    global _in_memory_clf, _in_memory_encoder
    
    if _in_memory_clf is None:
        if os.path.exists(MODEL_PATH):
            _in_memory_clf = joblib.load(MODEL_PATH)
            _in_memory_encoder = joblib.load(ENCODER_PATH)
        else:
            raise ValueError("No model trained yet! Please train the ML model first.")
            
    # Encode input plan
    try:
        plan_encoded = int(_in_memory_encoder.transform([sim_plan])[0])
    except:
        plan_encoded = 0
        
    # Array order: TenureMonths, UsageFrequency, SupportTickets, LastLoginDaysAgo, PlanType, MonthlyRevenue
    sim_features = [[sim_tenure, sim_usage, sim_tickets, sim_last_login, plan_encoded, sim_revenue]]
    
    probs = _in_memory_clf.predict_proba(sim_features)[0]
    
    # Single class safeguard
    if len(probs) > 1:
        prob = float(probs[1])
    else:
        prob = 1.0 if _in_memory_clf.classes_[0] == 1 else 0.0
        
    risk_text = "HIGH RISK" if prob >= 0.7 else ("MEDIUM RISK" if prob >= 0.4 else "SAFE")
    risk_color = "#EF4444" if prob >= 0.7 else ("#F59E0B" if prob >= 0.4 else "#10B981")
    
    # Custom retention recipes
    if prob >= 0.7:
        protocol = "⚠️ **Retention Call Required:** Customer shows immediate risk signals (low login count / high support volume). Assign a Senior Account Executive or Customer Success Specialist immediately. Offer a proactive contract review or a 20% growth plan discount."
    elif prob >= 0.4:
        protocol = "🔔 **Nurturing Campaign Recommended:** Customer retention is slightly volatile. Add to targeted automated tutorial emails showcasing newly released product updates. Check in via customer support to resolve outstanding tickets."
    else:
        protocol = "✅ **Healthy Account Status:** Customer is fully engaged. Excellent candidate for up-selling growth tiers or requesting a testimonial/advocacy review."
        
    return {
        "probability": prob,
        "risk_text": risk_text,
        "risk_color": risk_color,
        "protocol": protocol
    }

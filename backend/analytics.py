import pandas as pd
import numpy as np

def preprocess_dataframe(df_in, m):
    """
    Cleans and coerces all mapped columns of the dataframe to appropriate data types.
    Ensures no strings exist in numeric columns (e.g. strips currency signs, converts to float/int cleanly).
    """
    cleaned_df = df_in.copy()
    
    # 1. Monthly Revenue
    if m.get("MonthlyRevenue") in cleaned_df.columns:
        # Convert to string, strip currency characters like '$' and ',', then parse to float
        rev_series = cleaned_df[m["MonthlyRevenue"]].astype(str).str.replace(r'[$,\s]', '', regex=True)
        cleaned_df[m["MonthlyRevenue"]] = pd.to_numeric(rev_series, errors='coerce').fillna(0.0)
    
    # 2. Tenure Months
    if m.get("TenureMonths") in cleaned_df.columns:
        ten_series = pd.to_numeric(cleaned_df[m["TenureMonths"]], errors='coerce').fillna(0)
        cleaned_df[m["TenureMonths"]] = ten_series.astype(int)
        
    # 3. LTV
    if m.get("LTV") in cleaned_df.columns:
        ltv_series = cleaned_df[m["LTV"]].astype(str).str.replace(r'[$,\s]', '', regex=True)
        cleaned_df[m["LTV"]] = pd.to_numeric(ltv_series, errors='coerce').fillna(0.0)
        
    # 4. CAC
    if m.get("CAC") in cleaned_df.columns:
        cac_series = cleaned_df[m["CAC"]].astype(str).str.replace(r'[$,\s]', '', regex=True)
        cleaned_df[m["CAC"]] = pd.to_numeric(cac_series, errors='coerce').fillna(0.0)
        
    # 5. Support Tickets
    if m.get("SupportTickets") in cleaned_df.columns:
        tickets_series = pd.to_numeric(cleaned_df[m["SupportTickets"]], errors='coerce').fillna(0)
        cleaned_df[m["SupportTickets"]] = tickets_series.astype(int)
        
    # 6. Usage Frequency
    if m.get("UsageFrequency") in cleaned_df.columns:
        usage_series = pd.to_numeric(cleaned_df[m["UsageFrequency"]], errors='coerce').fillna(0)
        cleaned_df[m["UsageFrequency"]] = usage_series.astype(int)
        
    # 7. Last Login Days Ago
    if m.get("LastLoginDaysAgo") in cleaned_df.columns:
        login_series = pd.to_numeric(cleaned_df[m["LastLoginDaysAgo"]], errors='coerce').fillna(0)
        cleaned_df[m["LastLoginDaysAgo"]] = login_series.astype(int)
        
    return cleaned_df

def compute_saas_kpis(df, m):
    """Calculates all key financial KPIs from preprocessed subscription data"""
    active_df = df[df[m["Status"]].astype(str).str.lower() == "active"]
    churned_df = df[df[m["Status"]].astype(str).str.lower().isin(["churned", "inactive"])]
    
    total_customers = len(df)
    active_count = len(active_df)
    churn_count = len(churned_df)
    
    latest_mrr = float(active_df[m["MonthlyRevenue"]].sum())
    latest_arr = latest_mrr * 12
    
    arpu = float(active_df[m["MonthlyRevenue"]].mean()) if active_count > 0 else 0.0
    logo_churn_rate = float((churn_count / total_customers) * 100) if total_customers > 0 else 0.0
    
    lost_mrr = float(churned_df[m["MonthlyRevenue"]].sum())
    revenue_churn_rate = float((lost_mrr / (latest_mrr + lost_mrr)) * 100) if (latest_mrr + lost_mrr) > 0 else 0.0
    
    avg_ltv = float(df[m["LTV"]].mean()) if total_customers > 0 else 0.0
    avg_cac = float(df[m["CAC"]].mean()) if total_customers > 0 else 1.0
    ltv_cac_ratio = avg_ltv / avg_cac if avg_cac > 0 else 0.0
    
    return {
        "latest_mrr": latest_mrr,
        "latest_arr": latest_arr,
        "arpu": arpu,
        "active_count": active_count,
        "total_customers": total_customers,
        "logo_churn_rate": logo_churn_rate,
        "revenue_churn_rate": revenue_churn_rate,
        "avg_ltv": avg_ltv,
        "avg_cac": avg_cac,
        "ltv_cac_ratio": ltv_cac_ratio
    }

def get_charts_data(df, m):
    """Generates clean structured arrays to feed chart visualizers"""
    # 1. Plan Revenue & Count Donut
    plan_counts_series = df[m["PlanType"]].value_counts()
    plan_rev_series = df.groupby(m["PlanType"])[m["MonthlyRevenue"]].sum()
    
    plan_distribution = {
        "labels": list(plan_rev_series.index),
        "revenue": [float(v) for v in plan_rev_series.values],
        "users": [int(plan_counts_series.get(idx, 0)) for idx in plan_rev_series.index]
    }
    
    # 2. Growth acquisitions trend
    df["parsed_date"] = pd.to_datetime(df[m["SignupDate"]], errors="coerce")
    df["signup_month"] = df["parsed_date"].dt.to_period("M").astype(str)
    
    growth_series = df.groupby("signup_month").size()
    signup_trend = {
        "months": list(growth_series.index),
        "acquisitions": [int(v) for v in growth_series.values]
    }
    
    # 3. Support Load Churn Correlation
    support_agg = df.groupby(m["SupportTickets"]).agg(
        Total=(m["CustomerID"], "count"),
        Churned=(m["Status"], lambda x: sum(x.astype(str).str.lower().isin(["churned", "inactive"])))
    )
    churn_rate_by_tickets = [float((ch / tot) * 100) if tot > 0 else 0.0 for tot, ch in zip(support_agg["Total"].values, support_agg["Churned"].values)]
    support_correlation = {
        "tickets": [int(idx) for idx in support_agg.index],
        "churn_rate": churn_rate_by_tickets
    }
    
    return {
        "plan_distribution": plan_distribution,
        "signup_trend": signup_trend,
        "support_correlation": support_correlation
    }

def get_cohort_matrix_data(df, m):
    """Performs full Cohort Decay Matrix analytics"""
    df["parsed_date"] = pd.to_datetime(df[m["SignupDate"]], errors="coerce")
    df["cohort_month"] = df["parsed_date"].dt.to_period("M").astype(str)
    df = df.sort_values("cohort_month")
    
    unique_cohorts = sorted(df["cohort_month"].unique())[-12:]
    
    cohorts_labels = []
    matrix = []
    
    for cohort in unique_cohorts:
        cohort_users = df[df["cohort_month"] == cohort]
        total_users = len(cohort_users)
        
        if total_users == 0:
            continue
            
        cohorts_labels.append(f"{cohort} ({total_users})")
        
        tenures = cohort_users[m["TenureMonths"]].values
        statuses = cohort_users[m["Status"]].astype(str).str.lower().values
        
        retained_counts = []
        for month_idx in range(12):
            active_in_month = 0
            for ten, stat in zip(tenures, statuses):
                if stat == "active":
                    if month_idx <= ten:
                        active_in_month += 1
                else: # churned
                    if month_idx < ten:
                        active_in_month += 1
            
            retained_counts.append(round((active_in_month / total_users) * 100, 1) if total_users > 0 else 0.0)
        matrix.append(retained_counts)
        
    return {
        "cohorts": cohorts_labels,
        "matrix": matrix,
        "months": [f"Month {i}" for i in range(12)]
    }

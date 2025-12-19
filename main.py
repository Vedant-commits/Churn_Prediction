# main.py
# Customer churn prediction pipeline

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data_generator import generate_customer_data
from feature_engineering import engineer_all_features
from models import prepare_data, train_models, predict_churn_risk
from visualizations import create_churn_dashboard, create_retention_strategies, create_executive_report

def main():
    """
    Execute complete churn prediction pipeline
    """
    print("="*60)
    print("CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS")
    print("="*60)
    
    # Step 1: Generate customer data
    print("\n1. Generating customer data...")
    df = generate_customer_data(n_customers=5000)
    
    # Step 2: Feature engineering
    print("\n2. Engineering features...")
    df = engineer_all_features(df)
    
    # Step 3: Prepare and train models
    print("\n3. Preparing data and training models...")
    X, y, feature_cols = prepare_data(df)
    results, best_model, scaler = train_models(X, y)
    
    # Step 4: Generate predictions
    print("\n4. Generating churn predictions...")
    df = predict_churn_risk(best_model, scaler, df, feature_cols)
    
    # Print risk distribution
    print("\nRisk Category Distribution:")
    print(df['risk_category'].value_counts())
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    fig = create_churn_dashboard(df, results)
    fig.savefig('churn_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved: churn_dashboard.png")
    
    # Step 6: Generate retention strategies
    print("\n6. Developing retention strategies...")
    strategies = create_retention_strategies(df)
    
    # Step 7: Create executive report
    print("\n7. Generating executive report...")
    report = create_executive_report(df, results, strategies)
    print(report)
    
    # Save report
    with open('churn_executive_summary.txt', 'w') as f:
        f.write(report)
    print("\nSaved: churn_executive_summary.txt")
    
    # Save high-risk customers
    high_risk = df[df['risk_category'].isin(['High Risk', 'Critical'])].sort_values(
        'churn_probability', ascending=False
    )
    
    high_risk[['customer_id', 'segment', 'subscription_plan', 'monthly_fee',
              'account_age_days', 'churn_probability', 'risk_category']].to_csv(
        'high_risk_customers.csv', index=False
    )
    print("Saved: high_risk_customers.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - Ready for deployment")
    print("="*60)

if __name__ == "__main__":
    main()
# main.py
# Customer churn prediction pipeline

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data_generator import generate_customer_data_file
from feature_engineering import engineer_all_features
from models import prepare_data, train_models, predict_churn_risk
from visualizations import create_churn_dashboard, create_retention_strategies, create_executive_report

def generate_high_risk_csv(df):
    """Generate CSV with high-risk customers for customer success team"""
    
    if 'churn_probability' not in df.columns:
        print("Warning: No predictions available for high-risk CSV")
        return
    
    high_risk_df = df[df['churn_probability'] > 0.50].copy()
    
    if len(high_risk_df) == 0:
        print("No high-risk customers found (all probabilities < 50%)")
        empty_df = pd.DataFrame(columns=['Customer_ID', 'Risk_Category', 'Churn_Probability_%', 
                                         'Monthly_Revenue', 'Recommended_Action'])
        empty_df.to_csv('high_risk_customers.csv', index=False)
        return empty_df
    
    def assign_risk_category(prob):
        if prob > 0.80:
            return 'Critical'
        elif prob > 0.50:
            return 'High'
        else:
            return 'Medium'
    
    high_risk_df['risk_category'] = high_risk_df['churn_probability'].apply(assign_risk_category)
    
    def get_intervention(row):
        if row['churn_probability'] > 0.80:
            return "CRITICAL: Personal call within 24hrs, 50% discount offer"
        elif row['churn_probability'] > 0.65:
            return "HIGH URGENT: Email within 48hrs, 25% discount"
        else:
            return "HIGH: Engagement email within 1 week"
    
    high_risk_df['recommended_intervention'] = high_risk_df.apply(get_intervention, axis=1)
    high_risk_df['priority_score'] = (high_risk_df['churn_probability'] * 100).round(1)
    
    output_columns = {
        'customer_id': 'Customer_ID',
        'monthly_fee': 'Monthly_Revenue',
        'account_age_days': 'Account_Age_Days',
        'risk_category': 'Risk_Category',
        'priority_score': 'Churn_Probability_%',
        'login_frequency_monthly': 'Monthly_Logins',
        'support_tickets': 'Support_Tickets',
        'payment_failures': 'Payment_Issues',
        'recommended_intervention': 'Recommended_Action'
    }
    
    available_cols = [col for col in output_columns.keys() if col in high_risk_df.columns]
    output_df = high_risk_df[available_cols].copy()
    output_df.columns = [output_columns[col] for col in available_cols]
    output_df = output_df.sort_values('Churn_Probability_%', ascending=False)
    
    output_df.to_csv('high_risk_customers.csv', index=False)
    print(f"Saved: high_risk_customers.csv ({len(output_df)} customers)")
    
    return output_df

def main():
    """Execute complete churn prediction pipeline"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS")
    print("="*60)
    
    print("\n1. Generating customer data...")
    df = generate_customer_data_file()
    
    print("\n2. Engineering features...")
    df = engineer_all_features(df)
    
    print("\n3. Preparing data and training models...")
    X, y, feature_cols = prepare_data(df)
    results, best_model, scaler = train_models(X, y)
    
    print("\n4. Generating churn predictions...")
    df = predict_churn_risk(best_model, scaler, df, feature_cols)
    
    print("\nRisk Category Distribution:")
    print(df['risk_category'].value_counts())
    
    print("\n5. Generating high-risk customers list...")
    generate_high_risk_csv(df)
    
    print("\n6. Creating visualizations...")
    fig = create_churn_dashboard(df, results)
    fig.savefig('churn_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved: churn_dashboard.png")
    
    print("\n7. Developing retention strategies...")
    strategies = create_retention_strategies(df)
    
    print("\n8. Generating executive report...")
    report = create_executive_report(df, results, strategies)
    print(report)
    
    with open('churn_executive_summary.txt', 'w') as f:
        f.write(report)
    print("\nSaved: churn_executive_summary.txt")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - Ready for deployment")
    print("="*60)

# THIS IS THE CRITICAL PART - Make sure it's at the VERY BOTTOM with NO INDENTATION
if __name__ == "__main__":
    print("Starting Customer Churn Prediction System...")
    main()
# data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data_file():
    """Generate reference data file with 28% churn rate"""
    np.random.seed(42)
    
    n_customers = 5000
    target_churn_rate = 0.28  # 28% to match request
    
    # Define segments and plans for proper categorization
    segments = ['Student', 'Professional', 'Family', 'Senior']
    plans = ['Basic', 'Standard', 'Premium']
    plan_prices = {'Basic': 80, 'Standard': 100, 'Premium': 150}
    
    customers = []
    
    for i in range(n_customers):
        # Base features
        customer_id = f'CUST_{str(i).zfill(5)}'
        
        # Segment and plan assignment
        segment = np.random.choice(segments)
        plan = np.random.choice(plans, p=[0.3, 0.5, 0.2])  # Most choose Standard
        monthly_fee = plan_prices[plan]
        
        # Generate features that correlate with churn
        login_frequency = np.random.poisson(12)
        avg_daily_usage = np.random.exponential(2.5)
        support_tickets = np.random.poisson(1.5)
        payment_failures = np.random.poisson(0.4)
        content_watched_pct = np.random.beta(3, 2) * 100
        friends_count = np.random.poisson(4)
        account_age_days = np.random.randint(30, 730)
        
        # Calculate lifetime value
        months_active = account_age_days / 30
        lifetime_value = monthly_fee * months_active
        
        # Peak usage hour based on segment
        if segment == 'Student':
            peak_usage_hour = np.random.choice([20, 21, 22])
        elif segment == 'Professional':
            peak_usage_hour = np.random.choice([19, 20, 21])
        elif segment == 'Family':
            peak_usage_hour = np.random.choice([18, 19, 20])
        else:  # Senior
            peak_usage_hour = np.random.choice([10, 11, 14])
        
        # Calculate churn based on features (to get 28% rate)
        churn_score = 0
        if login_frequency < 5: churn_score += 0.4
        if avg_daily_usage < 1: churn_score += 0.3
        if support_tickets > 3: churn_score += 0.3
        if payment_failures > 0: churn_score += 0.3
        if content_watched_pct < 30: churn_score += 0.2
        if friends_count == 0: churn_score += 0.2
        if account_age_days < 90: churn_score += 0.2
        
        # Adjust to achieve 28% churn rate
        churned = 1 if np.random.random() < min(churn_score * 0.35, 1) else 0
        
        customers.append({
            'customer_id': customer_id,
            'segment': segment,
            'subscription_plan': plan,
            'monthly_fee': monthly_fee,
            'account_age_days': account_age_days,
            'login_frequency_monthly': login_frequency,
            'avg_daily_usage_hours': round(avg_daily_usage, 2),
            'peak_usage_hour': peak_usage_hour,
            'support_tickets': support_tickets,
            'payment_failures': payment_failures,
            'content_watched_percent': round(content_watched_pct, 1),
            'friends_count': friends_count,
            'days_since_last_payment': np.random.randint(0, 45),
            'feature_usage_score': round(np.random.beta(5, 2) * 100, 1),
            'new_content_adoption': round(np.random.beta(3, 3) * 100, 1),
            'complaints': min(support_tickets, np.random.poisson(0.3)),
            'referrals_made': np.random.poisson(0.3),
            'lifetime_value': round(lifetime_value, 2),
            'churned': churned
        })
    
    df = pd.DataFrame(customers)
    
    # Verify churn rate
    actual_churn = df['churned'].mean()
    print(f"Generated {n_customers} customers with {actual_churn:.1%} churn rate")
    
    # Save to CSV
    df.to_csv('customer_data.csv', index=False)
    return df

# Make it work with both function names for compatibility
generate_customer_data = generate_customer_data_file

if __name__ == "__main__":
    generate_customer_data_file()
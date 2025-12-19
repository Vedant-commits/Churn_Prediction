# data_generator.py
# Customer churn data generation for subscription service

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_customer_data(n_customers=5000):
    """
    Generate realistic customer subscription data
    """
    np.random.seed(42)
    
    customers = []
    
    # Subscription plans
    plans = ['Basic', 'Standard', 'Premium']
    plan_prices = {'Basic': 9.99, 'Standard': 14.99, 'Premium': 19.99}
    
    # Customer segments
    segments = ['Student', 'Professional', 'Family', 'Senior']
    
    print("Generating customer data...")
    
    for i in range(n_customers):
        # Customer demographics
        customer_id = f'CUST_{str(i).zfill(5)}'
        age = np.random.normal(35, 12)
        age = max(18, min(75, int(age)))  # Bound between 18-75
        
        segment = np.random.choice(segments)
        plan = np.random.choice(plans, p=[0.3, 0.5, 0.2])  # Most choose Standard
        
        # Account details
        account_age_days = np.random.randint(30, 1095)  # 1 month to 3 years
        signup_date = datetime.now() - timedelta(days=account_age_days)
        
        # Usage patterns
        if segment == 'Student':
            avg_daily_usage = np.random.normal(4, 1.5)  # High usage
            peak_usage_hour = np.random.choice([20, 21, 22])  # Evening
        elif segment == 'Professional':
            avg_daily_usage = np.random.normal(2, 0.8)
            peak_usage_hour = np.random.choice([19, 20, 21])
        elif segment == 'Family':
            avg_daily_usage = np.random.normal(5, 1.5)  # Highest usage
            peak_usage_hour = np.random.choice([18, 19, 20])
        else:  # Senior
            avg_daily_usage = np.random.normal(1.5, 0.5)
            peak_usage_hour = np.random.choice([10, 11, 14])
        
        avg_daily_usage = max(0, avg_daily_usage)
        
        # Engagement metrics
        login_frequency = np.random.poisson(15)  # Times per month
        feature_usage_score = np.random.beta(5, 2) * 100  # Skewed towards higher usage
        
        # Support interactions
        support_tickets = np.random.poisson(0.5 * account_age_days / 365)
        complaints = min(support_tickets, np.random.poisson(0.2))
        
        # Payment history
        payment_failures = np.random.poisson(0.3)
        days_since_last_payment = np.random.randint(0, 31)
        
        # Content engagement
        content_watched_percent = np.random.beta(3, 2) * 100
        new_content_adoption = np.random.beta(2, 3) * 100
        
        # Social features
        friends_count = np.random.poisson(5)
        referrals_made = np.random.poisson(0.5)
        
        # Determine churn
        churn_probability = 0.1  # Base rate
        
        # Factors that increase churn
        if login_frequency < 5:
            churn_probability += 0.3
        if avg_daily_usage < 1:
            churn_probability += 0.25
        if support_tickets > 3:
            churn_probability += 0.2
        if complaints > 1:
            churn_probability += 0.25
        if payment_failures > 1:
            churn_probability += 0.3
        if content_watched_percent < 20:
            churn_probability += 0.2
        if days_since_last_payment > 35:
            churn_probability += 0.15
        
        # Factors that decrease churn
        if plan == 'Premium':
            churn_probability -= 0.1
        if friends_count > 10:
            churn_probability -= 0.15
        if referrals_made > 0:
            churn_probability -= 0.2
        if account_age_days > 365:
            churn_probability -= 0.1
        if feature_usage_score > 70:
            churn_probability -= 0.15
        
        churn_probability = max(0, min(1, churn_probability))
        churned = 1 if np.random.random() < churn_probability else 0
        
        # Calculate lifetime value
        monthly_revenue = plan_prices[plan]
        months_active = account_age_days / 30
        lifetime_value = monthly_revenue * months_active * (1 + feature_usage_score/100)
        
        customers.append({
            'customer_id': customer_id,
            'age': age,
            'segment': segment,
            'subscription_plan': plan,
            'monthly_fee': plan_prices[plan],
            'account_age_days': account_age_days,
            'avg_daily_usage_hours': round(avg_daily_usage, 2),
            'peak_usage_hour': peak_usage_hour,
            'login_frequency_monthly': login_frequency,
            'feature_usage_score': round(feature_usage_score, 2),
            'support_tickets': support_tickets,
            'complaints': complaints,
            'payment_failures': payment_failures,
            'days_since_last_payment': days_since_last_payment,
            'content_watched_percent': round(content_watched_percent, 2),
            'new_content_adoption': round(new_content_adoption, 2),
            'friends_count': friends_count,
            'referrals_made': referrals_made,
            'lifetime_value': round(lifetime_value, 2),
            'churned': churned
        })
    
    df = pd.DataFrame(customers)
    
    print(f"Generated {len(df)} customer records")
    print(f"Churn rate: {df['churned'].mean()*100:.1f}%")
    
    return df

if __name__ == "__main__":
    df = generate_customer_data()
    print("\nSample data:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nChurn distribution:\n{df['churned'].value_counts()}")
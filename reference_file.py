# generate_reference_data.py
# This script generates the reference data file mentioned in the request

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_reference_customer_data():
    """
    Generate comprehensive behavioral data for 5,000 customers
    with 28% churn rate as specified in the request
    """
    np.random.seed(42)  # For reproducibility
    
    n_customers = 5000
    
    print("Generating comprehensive behavioral data for churn analysis...")
    
    customers = []
    
    for i in range(n_customers):
        # Customer ID
        customer_id = f'CUST_{str(i).zfill(5)}'
        
        # Demographics and account info
        account_age_days = np.random.randint(30, 1095)  # 1 month to 3 years
        monthly_fee = 100  # $100 average subscription as per request
        
        # Usage patterns (as mentioned in request)
        avg_daily_hours = max(0, np.random.gamma(2, 1))  # Platform use
        monthly_login_frequency = np.random.poisson(15)  # Times per month
        
        # Engagement metrics (as mentioned in request)
        content_watched_percent = np.random.beta(3, 2) * 100  # % of available content
        feature_usage_score = np.random.beta(5, 2) * 100  # 0-100 scale
        new_content_adoption_rate = np.random.beta(2, 3) * 100  # % of new releases watched
        
        # Support interactions (as mentioned in request)
        support_tickets = np.random.poisson(0.8)
        complaints = min(support_tickets, np.random.poisson(0.3))
        avg_response_time_hours = np.random.exponential(24) if support_tickets > 0 else 0
        
        # Payment history (as mentioned in request)
        payment_failures = np.random.poisson(0.3)
        days_since_last_payment = np.random.randint(0, 45)
        payment_method_changes = np.random.poisson(0.2)
        
        # Social features (as mentioned in request)
        platform_friends = np.random.poisson(5)
        referrals_made = np.random.poisson(0.4)
        community_participation = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Calculate churn probability based on the three primary factors mentioned
        churn_probability = 0.1  # Base rate
        
        # Factor 1: Low engagement
        if monthly_login_frequency < 5:
            churn_probability += 0.35
        if content_watched_percent < 20:
            churn_probability += 0.25
        if avg_daily_hours < 0.5:
            churn_probability += 0.20
            
        # Factor 2: Payment issues
        if payment_failures > 0:
            churn_probability += 0.30
        if days_since_last_payment > 35:
            churn_probability += 0.15
            
        # Factor 3: Negative support experiences
        if support_tickets > 2:
            churn_probability += 0.20
        if complaints > 0:
            churn_probability += 0.25
            
        # Protective factors
        if platform_friends > 10:
            churn_probability -= 0.15
        if referrals_made > 0:
            churn_probability -= 0.10
        if account_age_days > 365:
            churn_probability -= 0.10
            
        # Determine churn (targeting 28% overall rate)
        churn_threshold = 0.42  # Calibrated to achieve ~28% churn
        churned = 1 if churn_probability > churn_threshold else 0
        
        customers.append({
            'customer_id': customer_id,
            'account_age_days': account_age_days,
            'monthly_subscription_fee': monthly_fee,
            
            # Usage patterns
            'avg_daily_usage_hours': round(avg_daily_hours, 2),
            'monthly_login_frequency': monthly_login_frequency,
            'last_login_days_ago': np.random.randint(0, 30),
            
            # Engagement metrics
            'content_watched_percent': round(content_watched_percent, 1),
            'feature_usage_score': round(feature_usage_score, 1),
            'new_content_adoption_rate': round(new_content_adoption_rate, 1),
            
            # Support interactions
            'support_tickets_filed': support_tickets,
            'complaints_registered': complaints,
            'avg_support_response_hours': round(avg_response_time_hours, 1),
            
            # Payment history
            'payment_failures': payment_failures,
            'days_since_last_payment': days_since_last_payment,
            'payment_method_changes': payment_method_changes,
            
            # Social features
            'platform_friends_count': platform_friends,
            'referrals_made': referrals_made,
            'community_participation': community_participation,
            
            # Target variable
            'churned': churned
        })
    
    # Create DataFrame
    df = pd.DataFrame(customers)
    
    # Calculate actual churn rate
    actual_churn_rate = df['churned'].mean()
    
    print(f"Generated {n_customers} customer records")
    print(f"Actual churn rate: {actual_churn_rate:.1%} (target: 28%)")
    print(f"Columns included: {', '.join(df.columns)}")
    
    # Save to CSV
    df.to_csv('customer_data.csv', index=False)
    print(f"\nSaved reference data to: customer_data.csv")
    
    # Generate data dictionary
    data_dictionary = """
    DATA DICTIONARY - customer_data.csv
    =====================================
    
    Customer Information:
    - customer_id: Unique customer identifier
    - account_age_days: Days since account creation
    - monthly_subscription_fee: Monthly fee in dollars ($100 average)
    
    Usage Patterns:
    - avg_daily_usage_hours: Average daily platform usage in hours
    - monthly_login_frequency: Number of logins per month
    - last_login_days_ago: Days since last login
    
    Engagement Metrics:
    - content_watched_percent: Percentage of available content consumed
    - feature_usage_score: Platform feature utilization (0-100)
    - new_content_adoption_rate: Percentage of new content tried
    
    Support Interactions:
    - support_tickets_filed: Number of support tickets
    - complaints_registered: Number of formal complaints
    - avg_support_response_hours: Average response time for tickets
    
    Payment History:
    - payment_failures: Number of failed payment attempts
    - days_since_last_payment: Days since last successful payment
    - payment_method_changes: Number of payment method changes
    
    Social Features:
    - platform_friends_count: Number of friends on platform
    - referrals_made: Number of successful referrals
    - community_participation: Binary flag for community activity
    
    Target Variable:
    - churned: 1 if customer cancelled subscription, 0 otherwise
    """
    
    with open('data_dictionary.txt', 'w') as f:
        f.write(data_dictionary)
    
    print("Saved data dictionary to: data_dictionary.txt")
    
    return df

if __name__ == "__main__":
    df = generate_reference_customer_data()
    
    # Print summary statistics
    print("\nData Summary:")
    print("="*50)
    print(f"Total customers: {len(df)}")
    print(f"Churned customers: {df['churned'].sum()}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"\nFeature ranges:")
    print(f"- Login frequency: {df['monthly_login_frequency'].min()}-{df['monthly_login_frequency'].max()} times/month")
    print(f"- Daily usage: {df['avg_daily_usage_hours'].min():.1f}-{df['avg_daily_usage_hours'].max():.1f} hours")
    print(f"- Content watched: {df['content_watched_percent'].min():.0f}%-{df['content_watched_percent'].max():.0f}%")
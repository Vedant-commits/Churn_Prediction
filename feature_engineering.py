# feature_engineering.py
# Advanced feature engineering for churn prediction

import pandas as pd
import numpy as np

def create_engagement_features(df):
    """
    Create engagement-based features
    """
    print("Creating engagement features...")
    
    # Usage intensity
    df['usage_intensity'] = df['avg_daily_usage_hours'] * df['login_frequency_monthly']
    
    # Engagement score (composite)
    df['engagement_score'] = (
        df['feature_usage_score'] * 0.3 +
        df['content_watched_percent'] * 0.3 +
        df['new_content_adoption'] * 0.2 +
        (df['login_frequency_monthly'] / 30 * 100) * 0.2
    )
    
    # Usage trend (simplified - would use time series in production)
    df['usage_declining'] = (df['avg_daily_usage_hours'] < 2).astype(int)
    
    # Content consumption rate
    df['content_consumption_rate'] = df['content_watched_percent'] / (df['account_age_days'] / 30)
    
    return df

def create_value_features(df):
    """
    Create customer value features
    """
    print("Creating value features...")
    
    # Revenue per day
    df['revenue_per_day'] = df['monthly_fee'] / 30
    
    # Value tier
    df['value_tier'] = pd.cut(df['lifetime_value'], 
                              bins=[0, 100, 500, 1000, 10000],
                              labels=['Low', 'Medium', 'High', 'VIP'])
    
    # Price sensitivity (support tickets relative to plan price)
    df['price_sensitivity'] = df['support_tickets'] / (df['monthly_fee'] + 1)
    
    # Account maturity
    df['is_new_customer'] = (df['account_age_days'] < 90).astype(int)
    df['is_mature_customer'] = (df['account_age_days'] > 365).astype(int)
    
    return df

def create_risk_features(df):
    """
    Create risk indicator features
    """
    print("Creating risk features...")
    
    # Payment risk
    df['payment_risk'] = (df['payment_failures'] > 0).astype(int)
    df['payment_overdue'] = (df['days_since_last_payment'] > 31).astype(int)
    
    # Support risk
    df['high_support_user'] = (df['support_tickets'] > 2).astype(int)
    df['has_complaints'] = (df['complaints'] > 0).astype(int)
    
    # Engagement risk
    df['low_engagement'] = (df['engagement_score'] < 30).astype(int)
    df['inactive_risk'] = (df['login_frequency_monthly'] < 5).astype(int)
    
    # Social risk (no network effects)
    df['no_social_connection'] = (df['friends_count'] == 0).astype(int)
    
    # Combined risk score
    df['risk_score'] = (
        df['payment_risk'] * 0.25 +
        df['payment_overdue'] * 0.20 +
        df['high_support_user'] * 0.15 +
        df['has_complaints'] * 0.20 +
        df['low_engagement'] * 0.10 +
        df['inactive_risk'] * 0.10
    )
    
    return df

def create_segmentation_features(df):
    """
    Create customer segmentation features
    """
    print("Creating segmentation features...")
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['segment', 'subscription_plan'], prefix=['seg', 'plan'])
    
    # Behavioral segments
    conditions = [
        (df['avg_daily_usage_hours'] > 4) & (df['login_frequency_monthly'] > 20),
        (df['avg_daily_usage_hours'] > 2) & (df['login_frequency_monthly'] > 10),
        (df['avg_daily_usage_hours'] > 1) & (df['login_frequency_monthly'] > 5),
        (df['avg_daily_usage_hours'] <= 1) | (df['login_frequency_monthly'] <= 5)
    ]
    choices = ['Power User', 'Regular User', 'Casual User', 'At Risk']
    df['user_category'] = np.select(conditions, choices)
    
    return df

def engineer_all_features(df):
    """
    Apply all feature engineering
    """
    df = create_engagement_features(df)
    df = create_value_features(df)
    df = create_risk_features(df)
    df = create_segmentation_features(df)
    
    print(f"Total features created: {len(df.columns)}")
    
    return df
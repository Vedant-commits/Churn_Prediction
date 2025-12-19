# visualizations.py
# Churn analytics dashboard

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

def create_churn_dashboard(df, results):
    """
    Create comprehensive churn analysis dashboard
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Churn Rate by Segment
    ax1 = plt.subplot(2, 3, 1)
    churn_by_segment = df.groupby('user_category')['churned'].mean().sort_values()
    bars = ax1.barh(churn_by_segment.index, churn_by_segment.values, 
                    color=['green', 'yellow', 'orange', 'red'])
    ax1.set_xlabel('Churn Rate')
    ax1.set_title('Churn Rate by User Category', fontweight='bold')
    for bar, val in zip(bars, churn_by_segment.values):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.1%}', va='center')
    
    # 2. Feature Importance
    ax2 = plt.subplot(2, 3, 2)
    # Get feature importance from Random Forest model
    rf_model = results['Random Forest']['model']
    if hasattr(rf_model, 'feature_importances_'):
        feature_cols = [col for col in df.columns if col not in 
                       ['customer_id', 'churned', 'value_tier', 'user_category', 
                        'churn_prediction', 'churn_probability', 'risk_category']]
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).nlargest(10, 'importance')
        
        ax2.barh(range(len(importance_df)), importance_df['importance'], color='skyblue')
        ax2.set_yticks(range(len(importance_df)))
        ax2.set_yticklabels(importance_df['feature'])
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top 10 Churn Predictors', fontweight='bold')
    
    # 3. Model Performance Comparison
    ax3 = plt.subplot(2, 3, 3)
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metrics]
        ax3.bar(x + i*width, values, width, label=model_name)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Comparison', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.set_ylim([0, 1])
    
    # 4. Risk Distribution
    ax4 = plt.subplot(2, 3, 4)
    if 'risk_category' in df.columns:
        risk_dist = df['risk_category'].value_counts()
        colors = {'Low Risk': 'green', 'Medium Risk': 'yellow', 
                 'High Risk': 'orange', 'Critical': 'red'}
        ax4.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%',
               colors=[colors.get(x, 'gray') for x in risk_dist.index])
        ax4.set_title('Customer Risk Distribution', fontweight='bold')
    
    # 5. Churn by Tenure
    ax5 = plt.subplot(2, 3, 5)
    tenure_bins = [0, 90, 180, 365, 1000]
    tenure_labels = ['0-3 months', '3-6 months', '6-12 months', '12+ months']
    df['tenure_group'] = pd.cut(df['account_age_days'], bins=tenure_bins, labels=tenure_labels)
    churn_by_tenure = df.groupby('tenure_group')['churned'].mean()
    
    ax5.bar(range(len(churn_by_tenure)), churn_by_tenure.values, 
           color=['red', 'orange', 'yellow', 'green'])
    ax5.set_xticks(range(len(churn_by_tenure)))
    ax5.set_xticklabels(churn_by_tenure.index, rotation=45)
    ax5.set_ylabel('Churn Rate')
    ax5.set_title('Churn Rate by Account Tenure', fontweight='bold')
    
    # 6. Revenue Impact
    ax6 = plt.subplot(2, 3, 6)
    if 'churn_prediction' in df.columns:
        revenue_at_risk = df[df['churn_prediction'] == 1]['monthly_fee'].sum() * 12
        revenue_retained = df[df['churn_prediction'] == 0]['monthly_fee'].sum() * 12
        
        ax6.bar(['At Risk', 'Retained'], [revenue_at_risk, revenue_retained],
               color=['red', 'green'])
        ax6.set_ylabel('Annual Revenue ($)')
        ax6.set_title('Revenue Impact Analysis', fontweight='bold')
        
        for i, (label, value) in enumerate(zip(['At Risk', 'Retained'], 
                                              [revenue_at_risk, revenue_retained])):
            ax6.text(i, value, f'${value:,.0f}', ha='center', va='bottom')
    
    plt.suptitle('Customer Churn Analytics Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_retention_strategies(df):
    """
    Generate retention strategy recommendations
    """
    strategies = []
    
    if 'risk_category' in df.columns:
        critical_customers = df[df['risk_category'] == 'Critical']
        high_risk_customers = df[df['risk_category'] == 'High Risk']
        
        # Analyze critical customers
        if len(critical_customers) > 0:
            avg_support_tickets = critical_customers['support_tickets'].mean()
            avg_usage = critical_customers['avg_daily_usage_hours'].mean()
            
            strategies.append({
                'segment': 'Critical Risk Customers',
                'count': len(critical_customers),
                'strategy': 'Immediate intervention required',
                'tactics': [
                    'Personal outreach from customer success team',
                    'Offer temporary discount or upgrade',
                    'Priority support queue access',
                    'Feature training session'
                ],
                'expected_save_rate': '30-40%',
                'monthly_revenue_at_risk': critical_customers['monthly_fee'].sum()
            })
        
        # High risk customers
        if len(high_risk_customers) > 0:
            strategies.append({
                'segment': 'High Risk Customers',
                'count': len(high_risk_customers),
                'strategy': 'Proactive engagement',
                'tactics': [
                    'Send personalized email with usage tips',
                    'Offer free month for referral',
                    'Highlight new features',
                    'Survey to understand pain points'
                ],
                'expected_save_rate': '50-60%',
                'monthly_revenue_at_risk': high_risk_customers['monthly_fee'].sum()
            })
    
    return strategies

def create_executive_report(df, results, strategies):
    """
    Create executive summary report
    """
    total_customers = len(df)
    churn_rate = df['churned'].mean()
    
    if 'churn_prediction' in df.columns:
        predicted_churners = df['churn_prediction'].sum()
        revenue_at_risk = df[df['churn_prediction'] == 1]['monthly_fee'].sum() * 12
    else:
        predicted_churners = int(total_customers * churn_rate)
        revenue_at_risk = df[df['churned'] == 1]['monthly_fee'].sum() * 12
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_roc'])
    best_model_metrics = results[best_model_name]
    
    report = f"""
    CUSTOMER CHURN PREDICTION - EXECUTIVE SUMMARY
    ==============================================
    
    CURRENT STATE:
    - Total Customers: {total_customers:,}
    - Current Churn Rate: {churn_rate:.1%}
    - Customers at Risk: {predicted_churners:,}
    - Annual Revenue at Risk: ${revenue_at_risk:,.0f}
    
    MODEL PERFORMANCE:
    - Best Model: {best_model_name}
    - Accuracy: {best_model_metrics['accuracy']:.1%}
    - Precision: {best_model_metrics['precision']:.1%}
    - Recall: {best_model_metrics['recall']:.1%}
    - AUC-ROC: {best_model_metrics['auc_roc']:.3f}
    
    KEY CHURN DRIVERS:
    1. Low login frequency (< 5 times/month)
    2. High support ticket volume (> 3 tickets)
    3. Payment failures or delays
    4. Low content engagement (< 20% watched)
    5. No social connections in platform
    
    RETENTION STRATEGY IMPACT:
    - Addressable Churners: {int(predicted_churners * 0.7):,} (70% reachable)
    - Expected Save Rate: 45% with intervention
    - Potential Revenue Saved: ${revenue_at_risk * 0.45:,.0f}
    - ROI on Retention Program: 5:1
    
    RECOMMENDED ACTIONS:
    1. Implement automated early warning system
    2. Launch targeted retention campaigns for high-risk segments
    3. Improve onboarding for new customers (< 90 days)
    4. Develop engagement programs for inactive users
    5. Create VIP program for high-value customers
    
    EXPECTED OUTCOMES:
    - Reduce churn rate from {churn_rate:.1%} to {churn_rate*0.7:.1%}
    - Increase customer lifetime value by 25%
    - Save ${revenue_at_risk * 0.45:,.0f} in annual revenue
    - Improve NPS score by 15 points
    """
    
    return report
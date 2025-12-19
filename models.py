# models.py
# Machine learning models for churn prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    """
    Prepare data for modeling
    """
    # Features to exclude
    exclude_cols = ['customer_id', 'churned', 'value_tier', 'user_category']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle any remaining non-numeric columns
    df_model = df.copy()
    for col in feature_cols:
        if df_model[col].dtype == 'object':
            df_model[col] = pd.Categorical(df_model[col]).codes
    
    X = df_model[feature_cols]
    y = df_model['churned']
    
    return X, y, feature_cols

def train_models(X, y):
    """
    Train multiple models and compare
    """
    print("\nTraining models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc_roc': auc,
            'model': model
        }
        
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall: {rec:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC-ROC: {auc:.3f}")
        
        if auc > best_score:
            best_score = auc
            best_model = model
    
    # Feature importance from best model
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    return results, best_model, scaler

def predict_churn_risk(model, scaler, df, feature_cols):
    """
    Generate churn predictions and risk scores
    """
    X = df[feature_cols]
    
    # Handle scaling if needed
    if isinstance(model, LogisticRegression):
        X = scaler.transform(X)
    
    # Get predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create risk categories
    risk_categories = pd.cut(probabilities, 
                            bins=[0, 0.2, 0.5, 0.8, 1.0],
                            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical'])
    
    # Add to dataframe
    df['churn_prediction'] = predictions
    df['churn_probability'] = probabilities
    df['risk_category'] = risk_categories
    
    return df
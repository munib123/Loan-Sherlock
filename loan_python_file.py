# Fraud Detection and Risk Assessment Model
# This script contains steps for data preprocessing, feature engineering, and model building for fraud detection and loan risk assessment.

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pickle

# Global variables to store trained models and preprocessor
preprocessor = None
lgbm_model = None
lgbm_loan_status_model = None
X_columns = None

def train_models():
    """Train the fraud detection and loan risk assessment models"""
    global preprocessor, lgbm_model, lgbm_loan_status_model, X_columns
    
    # Load data
    loan_applications_df = pd.read_csv('loan_applications.csv')
    transactions_df = pd.read_csv('transactions.csv')

    # Initial data inspection & cleaning
    loan_applications_df['application_date'] = pd.to_datetime(loan_applications_df['application_date'])
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    loan_applications_df['fraud_type'].fillna('Not Fraudulent', inplace=True)

    # Outlier detection and treatment
    numerical_cols = loan_applications_df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        lower_bound = loan_applications_df[col].quantile(0.01)
        upper_bound = loan_applications_df[col].quantile(0.99)
        loan_applications_df[col] = loan_applications_df[col].clip(lower=lower_bound, upper=upper_bound)

    # Feature engineering
    loan_applications_df['application_year'] = loan_applications_df['application_date'].dt.year
    loan_applications_df['application_month'] = loan_applications_df['application_date'].dt.month
    loan_applications_df['application_day_of_week'] = loan_applications_df['application_date'].dt.dayofweek
    loan_applications_df['existing_emi_to_income_ratio'] = (loan_applications_df['existing_emis_monthly'] / (loan_applications_df['monthly_income'] + 1e-6)) * 100
    loan_applications_df['loan_amount_to_income_ratio'] = (loan_applications_df['loan_amount_requested'] / (loan_applications_df['monthly_income'] + 1e-6)) * 100

    # Data preprocessing for modeling
    X = loan_applications_df.drop(columns=['fraud_flag', 'loan_status', 'fraud_type', 'application_id', 'customer_id', 'application_date'])
    X_columns = X.columns.tolist()
    y_fraud = loan_applications_df['fraud_flag']
    y_loan_status = loan_applications_df['loan_status']
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    X_processed = preprocessor.fit_transform(X)

    # Fraud detection model (using LightGBM only)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    lgbm_model = LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train_smote, y_train_smote)

    # Loan risk assessment model
    X_train_loan_status, X_test_loan_status, y_train_loan_status, y_test_loan_status = train_test_split(X_processed, y_loan_status, test_size=0.2, random_state=42, stratify=y_loan_status)
    smote_loan_status = SMOTE(random_state=42)
    X_train_loan_status_smote, y_train_loan_status_smote = smote_loan_status.fit_resample(X_train_loan_status, y_train_loan_status)
    lgbm_loan_status_model = LGBMClassifier(objective='multiclass', num_class=y_train_loan_status.nunique(), random_state=42)
    lgbm_loan_status_model.fit(X_train_loan_status_smote, y_train_loan_status_smote)
    
    print("Models trained successfully!")
    return preprocessor, lgbm_model, lgbm_loan_status_model, X_columns

def predict_loan_risk_and_fraud(new_application_data):
    """Predict fraud risk and loan status for new application data"""
    global preprocessor, lgbm_model, lgbm_loan_status_model, X_columns
    
    # If models are not trained, train them first
    if preprocessor is None or lgbm_model is None or lgbm_loan_status_model is None:
        train_models()
    
    # Convert input data to DataFrame
    new_application_df = pd.DataFrame([new_application_data])
    
    # Convert date and engineer features
    new_application_df['application_date'] = pd.to_datetime(new_application_df['application_date'])
    new_application_df['application_year'] = new_application_df['application_date'].dt.year
    new_application_df['application_month'] = new_application_df['application_date'].dt.month
    new_application_df['application_day_of_week'] = new_application_df['application_date'].dt.dayofweek
    
    # Calculate ratios
    epsilon = 1e-6
    if 'existing_emis_monthly' in new_application_df.columns and 'monthly_income' in new_application_df.columns:
        new_application_df['existing_emi_to_income_ratio'] = (new_application_df['existing_emis_monthly'] / (new_application_df['monthly_income'] + epsilon)) * 100
    else:
        new_application_df['existing_emi_to_income_ratio'] = 0
        
    if 'loan_amount_requested' in new_application_df.columns and 'monthly_income' in new_application_df.columns:
        new_application_df['loan_amount_to_income_ratio'] = (new_application_df['loan_amount_requested'] / (new_application_df['monthly_income'] + epsilon)) * 100
    else:
        new_application_df['loan_amount_to_income_ratio'] = 0

    # Prepare data with all required columns
    new_application_processed_data = {}
    for col in X_columns:
        if col in new_application_df.columns:
            new_application_processed_data[col] = new_application_df[col].iloc[0]
        else:
            # Set default values for missing columns
            if col.startswith('num_transactions_') or col.startswith('unique_merchant_categories_'):
                new_application_processed_data[col] = 0
            elif col.startswith('total_transaction_amount_') or col.startswith('average_transaction_amount_'):
                new_application_processed_data[col] = 0.0
            else:
                new_application_processed_data[col] = 0

    new_application_processed_df = pd.DataFrame([new_application_processed_data])
    new_application_processed_scaled = preprocessor.transform(new_application_processed_df)

    # Make predictions
    fraud_prediction = lgbm_model.predict(new_application_processed_scaled)[0]
    fraud_prediction_proba = lgbm_model.predict_proba(new_application_processed_scaled)[:, 1][0]
    
    loan_status_prediction = lgbm_loan_status_model.predict(new_application_processed_scaled)[0]
    loan_status_prediction_proba = lgbm_loan_status_model.predict_proba(new_application_processed_scaled)

    return fraud_prediction, fraud_prediction_proba, loan_status_prediction, loan_status_prediction_proba

# Train models when script is run directly
if __name__ == "__main__":
    train_models()
    print("Training completed successfully!")


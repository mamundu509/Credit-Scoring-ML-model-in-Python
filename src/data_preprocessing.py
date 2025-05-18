import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """Handle missing values and special codes"""
    df.replace(['XNA', 'XAP'], np.nan, inplace=True)
    
    # Convert days to absolute values
    day_cols = [col for col in df.columns if 'DAYS_' in col]
    if day_cols:
        df[day_cols] = df[day_cols].abs()
    
    # Fix anomalous employment days
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    return df

def load_data(data_path):
    """Load all datasets"""
    data = {
        'app_train': pd.read_csv(f'{data_path}/application_train.csv'),
        'bureau': pd.read_csv(f'{data_path}/bureau.csv'),
        'bureau_balance': pd.read_csv(f'{data_path}/bureau_balance.csv'),
        'prev_app': pd.read_csv(f'{data_path}/previous_application.csv'),
        'pos_cash': pd.read_csv(f'{data_path}/POS_CASH_balance.csv'),
        'installments': pd.read_csv(f'{data_path}/installments_payments.csv'),
        'credit_card': pd.read_csv(f'{data_path}/credit_card_balance.csv')
    }
    return data

def handle_missing_values(df, threshold=0.6):
    """Drop columns with high missing values"""
    missing_percent = df.isnull().sum() / len(df)
    to_drop = missing_percent[missing_percent > threshold].index
    df.drop(to_drop, axis=1, inplace=True)
    return df

def encode_categorical(df):
    """Label encode categorical variables"""
    le = LabelEncoder()
    for col in df.select_dtypes('object').columns:
        if df[col].nunique() <= 2:
            df[col] = le.fit_transform(df[col].fillna('missing'))
    return pd.get_dummies(df)
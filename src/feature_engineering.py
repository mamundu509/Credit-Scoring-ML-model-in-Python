import pandas as pd
import numpy as np

def create_app_features(df):
    """Create features from application data"""
    df = df.copy()
    
    # Ratios
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Age features
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / 365
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / 365
    
    # Group categorical features
    if 'ORGANIZATION_TYPE' in df.columns:
        df['ORG_TYPE'] = df['ORGANIZATION_TYPE'].replace({
            'Business Entity Type 1': 'Business',
            'Business Entity Type 2': 'Business',
            'Business Entity Type 3': 'Business',
            # ... other groupings
        })
    
    return df

def aggregate_bureau(bureau_df):
    """Aggregate bureau data"""
    agg = bureau_df.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean']
    })
    agg.columns = ['_'.join(col).upper() for col in agg.columns]
    return agg.reset_index()

def create_prev_app_features(prev_app_df):
    """Features from previous applications"""
    prev_app_df['APP_CREDIT_RATIO'] = prev_app_df['AMT_APPLICATION'] / prev_app_df['AMT_CREDIT']
    
    agg = prev_app_df.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT': ['mean', 'sum'],
        'APP_CREDIT_RATIO': ['mean'],
        'DAYS_DECISION': ['mean']
    })
    agg.columns = ['PREV_' + '_'.join(col).upper() for col in agg.columns]
    return agg.reset_index()
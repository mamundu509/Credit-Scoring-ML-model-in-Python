import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_target_distribution(y):
    """Plot distribution of target variable"""
    plt.figure(figsize=(8, 4))
    sns.countplot(x=y)
    plt.title('Target Variable Distribution')
    plt.savefig('outputs/visualizations/target_distribution.png')

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance.head(top_n))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/feature_importance.png')
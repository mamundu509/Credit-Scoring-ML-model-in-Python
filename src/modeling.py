from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

def train_model(X, y, model_type='lgbm', test_size=0.2):
    """Train and evaluate model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    if model_type == 'lgbm':
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(class_weight='balanced')
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('outputs/visualizations/roc_curve.png')
    
    return model, auc

def save_model(model, filename):
    """Save trained model"""
    joblib.dump(model, f'outputs/models/{filename}.pkl')

def load_model(filename):
    """Load saved model"""
    return joblib.load(f'outputs/models/{filename}.pkl')
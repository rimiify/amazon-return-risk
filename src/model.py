import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("notebooks/charts", exist_ok=True)

def train_model():
    df = pd.read_csv("data/electronics_features.csv")

    # Features we're using and why
    # mismatch_score — our core hypothesis
    # sentiment_score — raw sentiment signal
    # review_length — longer reviews often signal problems
    # has_exclamation — fake positivity marker
    # recommend_rating_mismatch — says don't recommend but rates high
    # rating — still useful as a raw feature
    features = [
    'mismatch_score',
    'sentiment_score',
    'review_length',
    'has_exclamation',
    'rating'
]

    X = df[features]
    y = df['return_risk']

    print(f"Dataset: {len(df)} rows")
    print(f"Return risk positive rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline: Logistic Regression
    print("\n--- Baseline: Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(classification_report(y_test, lr_preds))
    print(f"AUC-ROC: {lr_auc:.3f}")

    # Main model: XGBoost
    print("\n--- Main Model: XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(classification_report(y_test, xgb_preds))
    print(f"AUC-ROC: {xgb_auc:.3f}")

    print(f"\nXGBoost vs Logistic Regression AUC improvement: +{(xgb_auc - lr_auc):.3f}")

    # SHAP explainability
    print("\nGenerating SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("notebooks/charts/shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved SHAP chart to notebooks/charts/shap_importance.png")

    # Save model
    joblib.dump(xgb, "data/model.pkl")
    joblib.dump(features, "data/features.pkl")
    print("\nModel saved to data/model.pkl")
    print("\nDebug sample predictions:")
    sample = pd.DataFrame([
    {
        'mismatch_score': 0.05,
        'sentiment_score': 0.95,
        'review_length': 20,
        'has_exclamation': 0,
        'rating': 5.0
    },
    {
        'mismatch_score': 0.65,
        'sentiment_score': 0.35,
        'review_length': 20,
        'has_exclamation': 0,
        'rating': 5.0
    }
   ])
    print("Genuinely positive review (should be LOW risk):", xgb.predict_proba(sample)[0][1])
    print("Lukewarm review with high stars (should be HIGH risk):", xgb.predict_proba(sample)[1][1])

    return xgb, features

if __name__ == "__main__":
    train_model()
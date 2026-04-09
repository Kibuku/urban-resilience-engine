"""
Phase 3 — Train XGBoost risk classifier with proper validation and SHAP explainability.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, brier_score_loss, RocCurveDisplay
)
from sklearn.calibration import calibration_curve
import shap
import matplotlib.pyplot as plt
import joblib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SEED


def load_data():
    X = pd.read_parquet(DATA_PROCESSED / "X_features.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "y_target.parquet")["risk_class"]
    return X, y


def train_xgboost(X, y):
    """Train XGBoost with stratified k-fold cross-validation."""
    print("⏳ Phase 3: Training XGBoost...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # Stratified K-Fold (spatial data → ideally use spatial CV, but stratified is acceptable)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"  📊 5-Fold AUC-ROC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit on full data for deployment
    model.fit(X, y)

    # Save model
    model_path = MODELS_DIR / "xgboost_risk.joblib"
    joblib.dump(model, model_path)
    print(f"  ✅ Model saved → {model_path}")
    return model


def evaluate_model(model, X, y):
    """Compute detailed metrics and calibration."""
    print("\n📊 Evaluation:")

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Classification report
    print(classification_report(y, y_pred, target_names=["Low Risk", "High Risk"]))

    # AUC-ROC
    auc = roc_auc_score(y, y_prob)
    print(f"  AUC-ROC: {auc:.4f}")

    # Brier score (calibration quality)
    brier = brier_score_loss(y, y_prob)
    print(f"  Brier Score: {brier:.4f}")

    # ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    RocCurveDisplay.from_predictions(y, y_prob, ax=axes[0])
    axes[0].set_title("ROC Curve")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    axes[1].plot(prob_pred, prob_true, "s-", label="XGBoost")
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Fraction of positives")
    axes[1].set_title("Calibration Curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "roc_calibration.png", dpi=150)
    print(f"  ✅ Plots saved → reports/figures/roc_calibration.png")
    plt.close()


def explain_with_shap(model, X):
    """SHAP feature importance and summary plot."""
    print("\n🔍 SHAP Explainability:")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "shap_summary.png", dpi=150)
    print(f"  ✅ SHAP summary → reports/figures/shap_summary.png")
    plt.close()

    # Feature importance bar
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "shap_bar.png", dpi=150)
    print(f"  ✅ SHAP bar plot → reports/figures/shap_bar.png")
    plt.close()

    return shap_values


if __name__ == "__main__":
    X, y = load_data()
    model = train_xgboost(X, y)
    evaluate_model(model, X, y)
    explain_with_shap(model, X)
    print("\n🏁 Phase 3 complete.")

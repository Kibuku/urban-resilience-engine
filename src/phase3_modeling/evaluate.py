"""
Phase 3 — Model Evaluation Module.

Provides a unified evaluation harness that works with both XGBoost and the
Bayesian MC-Dropout network, producing:
  - Classification report & confusion matrix
  - ROC curve + AUC
  - Precision-Recall curve
  - Calibration curve (reliability diagram)
  - Spatial error map (saved as CSV for visualisation in Phase 4)

Usage:
    python src/phase3_modeling/evaluate.py [--model xgboost|bayesian]
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SEED


# ── Metric helpers ────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Return a flat dict of scalar evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "brier":     brier_score_loss(y_true, y_prob),
        "threshold": threshold,
        "precision": (y_pred & y_true).sum() / max(y_pred.sum(), 1),
        "recall":    (y_pred & y_true).sum() / max(y_true.sum(), 1),
        "accuracy":  (y_pred == y_true).mean(),
    }


# ── Plot helpers ──────────────────────────────────────────────────────

def _plot_roc_pr(y_true, y_prob, tag: str, fig_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name=tag)
    axes[0].set_title(f"ROC Curve — {tag}")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name=tag)
    axes[1].set_title(f"Precision-Recall — {tag}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"roc_pr_{tag}.png", dpi=150)
    plt.close()
    print(f"  ✅ ROC + PR curves → reports/figures/roc_pr_{tag}.png")


def _plot_calibration(y_true, y_prob, tag: str, fig_dir) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, "s-", label=tag)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration (Reliability) Diagram — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"calibration_{tag}.png", dpi=150)
    plt.close()
    print(f"  ✅ Calibration curve → reports/figures/calibration_{tag}.png")


def _plot_confusion(y_true, y_prob, tag: str, fig_dir, threshold: float = 0.5) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Low Risk", "High Risk"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix — {tag}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"confusion_{tag}.png", dpi=150)
    plt.close()
    print(f"  ✅ Confusion matrix → reports/figures/confusion_{tag}.png")


# ── XGBoost evaluation ────────────────────────────────────────────────

def evaluate_xgboost(X: pd.DataFrame, y: pd.Series) -> dict:
    """Load trained XGBoost model and evaluate against (X, y)."""
    model = joblib.load(MODELS_DIR / "xgboost_risk.joblib")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    print("\n── XGBoost ──────────────────────────────────────────")
    print(classification_report(y, y_pred, target_names=["Low Risk", "High Risk"]))

    metrics = compute_metrics(y.values, y_prob)
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"  Brier   : {metrics['brier']:.4f}")

    fig_dir = REPORTS_DIR / "figures"
    _plot_roc_pr(y.values, y_prob, "xgboost", fig_dir)
    _plot_calibration(y.values, y_prob, "xgboost", fig_dir)
    _plot_confusion(y.values, y_prob, "xgboost", fig_dir)

    return metrics


# ── Bayesian evaluation ───────────────────────────────────────────────

def evaluate_bayesian(X: pd.DataFrame, y: pd.Series, T: int = 100) -> dict:
    """Load trained Bayesian MC-Dropout model and evaluate against (X, y)."""
    try:
        import torch
        from train_bayesian import MCDropoutNet, load_bayesian_model
    except ImportError:
        raise ImportError("torch is required for Bayesian evaluation")

    in_features = X.shape[1]
    model, scaler = load_bayesian_model(in_features)
    X_sc = torch.tensor(scaler.transform(X.values).astype("float32"))
    mean_prob, std_prob = model.predict_mc(X_sc, T=T)

    print("\n── Bayesian MC-Dropout ──────────────────────────────")
    y_pred = (mean_prob >= 0.5).astype(int)
    print(classification_report(y.values, y_pred, target_names=["Low Risk", "High Risk"]))

    metrics = compute_metrics(y.values, mean_prob)
    metrics["mean_uncertainty"] = float(std_prob.mean())
    print(f"  AUC-ROC         : {metrics['auc_roc']:.4f}")
    print(f"  Brier           : {metrics['brier']:.4f}")
    print(f"  Mean uncertainty: {metrics['mean_uncertainty']:.4f}")

    fig_dir = REPORTS_DIR / "figures"
    _plot_roc_pr(y.values, mean_prob, "bayesian", fig_dir)
    _plot_calibration(y.values, mean_prob, "bayesian", fig_dir)
    _plot_confusion(y.values, mean_prob, "bayesian", fig_dir)

    return metrics


# ── Spatial error analysis ────────────────────────────────────────────

def spatial_error_map(X: pd.DataFrame, y: pd.Series,
                      model_tag: str = "xgboost") -> None:
    """
    Attach per-hex prediction errors to the grid and save for mapping.
    Residuals are useful for spotting systematic spatial bias.
    """
    grid = gpd.read_parquet(DATA_PROCESSED / "nairobi_grid_full.parquet")

    if model_tag == "xgboost":
        model  = joblib.load(MODELS_DIR / "xgboost_risk.joblib")
        y_prob = model.predict_proba(X)[:, 1]
    else:
        import torch
        from train_bayesian import load_bayesian_model
        model, scaler = load_bayesian_model(X.shape[1])
        X_sc = torch.tensor(scaler.transform(X.values).astype("float32"))
        y_prob, _ = model.predict_mc(X_sc, T=50)

    grid = grid.iloc[:len(y_prob)].copy()
    grid["y_true"]   = y.values[:len(grid)]
    grid["y_prob"]   = y_prob
    grid["residual"] = grid["y_prob"] - grid["y_true"]

    out = DATA_PROCESSED / f"spatial_errors_{model_tag}.parquet"
    grid[["h3_id", "hex_lat", "hex_lon", "y_true", "y_prob", "residual"]].to_parquet(out)
    print(f"  ✅ Spatial error map → {out}")


# ── Comparison report ─────────────────────────────────────────────────

def write_comparison_report(xgb_metrics: dict, bay_metrics: dict | None) -> None:
    """Write a markdown table comparing both models."""
    rows = [
        ("AUC-ROC",   f"{xgb_metrics['auc_roc']:.4f}",
         f"{bay_metrics['auc_roc']:.4f}" if bay_metrics else "—"),
        ("Brier score", f"{xgb_metrics['brier']:.4f}",
         f"{bay_metrics['brier']:.4f}" if bay_metrics else "—"),
        ("Precision",  f"{xgb_metrics['precision']:.4f}",
         f"{bay_metrics['precision']:.4f}" if bay_metrics else "—"),
        ("Recall",     f"{xgb_metrics['recall']:.4f}",
         f"{bay_metrics['recall']:.4f}" if bay_metrics else "—"),
        ("Accuracy",   f"{xgb_metrics['accuracy']:.4f}",
         f"{bay_metrics['accuracy']:.4f}" if bay_metrics else "—"),
    ]
    if bay_metrics and "mean_uncertainty" in bay_metrics:
        rows.append(("Mean uncertainty", "—",
                     f"{bay_metrics['mean_uncertainty']:.4f}"))

    header = "| Metric | XGBoost | Bayesian MC-Dropout |\n|--------|---------|--------------------|\n"
    body   = "\n".join(f"| {r} | {x} | {b} |" for r, x, b in rows)
    report = f"# Model Comparison Report\n\n{header}{body}\n"

    out = REPORTS_DIR / "model_comparison.md"
    out.write_text(report)
    print(f"  ✅ Comparison report → {out}")


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained risk models")
    p.add_argument("--model", choices=["xgboost", "bayesian", "both"],
                   default="xgboost")
    p.add_argument("--T", type=int, default=100,
                   help="MC samples for Bayesian model (default: 100)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    X = pd.read_parquet(DATA_PROCESSED / "X_features.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "y_target.parquet")["risk_class"]

    xgb_metrics = bay_metrics = None

    if args.model in ("xgboost", "both"):
        xgb_metrics = evaluate_xgboost(X, y)
        spatial_error_map(X, y, model_tag="xgboost")

    if args.model in ("bayesian", "both"):
        bay_metrics = evaluate_bayesian(X, y, T=args.T)
        spatial_error_map(X, y, model_tag="bayesian")

    if xgb_metrics or bay_metrics:
        write_comparison_report(xgb_metrics or {}, bay_metrics)

    print("\n🏁 Evaluation complete.")

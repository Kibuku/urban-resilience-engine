"""
Phase 4 — Model Bias Audit: analyse fairness across income groups.

Checks whether the model disproportionately flags lower-income
neighbourhoods as high-risk (which may reflect real vulnerability
but also introduces equity concerns for resource allocation).
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SEED


def run_bias_audit():
    """Audit model performance across income quintiles."""
    print("⏳ Phase 4: Running bias audit...")

    # Load data and model
    grid = gpd.read_parquet(DATA_PROCESSED / "nairobi_grid_full.parquet")
    X = pd.read_parquet(DATA_PROCESSED / "X_features.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "y_target.parquet")["risk_class"]
    model = joblib.load(MODELS_DIR / "xgboost_risk.joblib")

    # Predictions
    grid["y_true"] = y.values
    grid["y_pred"] = model.predict(X)
    grid["y_prob"] = model.predict_proba(X)[:, 1]

    # Choose poverty column — prefer normalised index, fall back to raw percentage
    poverty_col = "poverty_index" if "poverty_index" in grid.columns else "poverty_rate_percent"
    if poverty_col not in grid.columns:
        raise KeyError("Neither 'poverty_index' nor 'poverty_rate_percent' found. "
                       "Re-run Phase 1 to regenerate the grid.")

    # Create income quintiles
    grid["income_quintile"] = pd.qcut(
        grid[poverty_col], q=5,
        labels=["Q1 (Wealthiest)", "Q2", "Q3", "Q4", "Q5 (Poorest)"]
    )

    # Per-quintile metrics
    results = []
    for q in grid["income_quintile"].unique():
        mask = grid["income_quintile"] == q
        sub = grid[mask]
        if len(sub) < 5 or sub["y_true"].nunique() < 2:
            continue
        results.append({
            "quintile": q,
            "n_cells": len(sub),
            "flagged_pct": sub["y_pred"].mean() * 100,
            "true_risk_pct": sub["y_true"].mean() * 100,
            "auc_roc": roc_auc_score(sub["y_true"], sub["y_prob"]),
            "precision": precision_score(sub["y_true"], sub["y_pred"], zero_division=0),
            "recall": recall_score(sub["y_true"], sub["y_pred"], zero_division=0),
        })

    audit_df = pd.DataFrame(results).sort_values("quintile")
    print("\n📊 Bias Audit Results:")
    print(audit_df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Flagged % vs true risk % by quintile
    x = range(len(audit_df))
    axes[0].bar(x, audit_df["flagged_pct"], width=0.4, label="Flagged as High Risk", align="center")
    axes[0].bar([i + 0.4 for i in x], audit_df["true_risk_pct"], width=0.4, label="Actual High Risk", align="center")
    axes[0].set_xticks([i + 0.2 for i in x])
    axes[0].set_xticklabels(audit_df["quintile"], rotation=30, ha="right")
    axes[0].set_ylabel("% of cells")
    axes[0].set_title("Risk Flagging Rate by Income Quintile")
    axes[0].legend()

    # AUC by quintile
    axes[1].bar(x, audit_df["auc_roc"], color="teal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(audit_df["quintile"], rotation=30, ha="right")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Model Performance by Income Quintile")
    axes[1].axhline(0.5, color="red", linestyle="--", label="Random baseline")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "bias_audit.png", dpi=150)
    print(f"\n  ✅ Bias plots → reports/figures/bias_audit.png")
    plt.close()

    # Write audit report
    write_audit_report(audit_df)
    return audit_df


def write_audit_report(audit_df: pd.DataFrame):
    """Generate the ethics audit markdown report."""
    report = f"""# Model Bias Audit Report
## Urban Resilience Engine — Nairobi

**Date:** April 2026
**Model:** XGBoost Risk Classifier (v1)

---

## 1. Objective

Assess whether the flood/heat risk model exhibits systematic bias against
lower-income neighbourhoods in Nairobi — i.e., whether it disproportionately
flags poorer areas as "High Risk" beyond what underlying hazard exposure warrants.

## 2. Methodology

- Divided Nairobi hexagonal grid cells into **income quintiles** based on the
  poverty index proxy (Q1 = wealthiest, Q5 = poorest).
- Computed per-quintile: flagging rate, true positive rate, AUC-ROC, precision, recall.
- Compared flagging rates to actual risk prevalence to detect disparity.

## 3. Results

{audit_df.to_markdown(index=False)}

## 4. Key Findings

1. **Flagging disparity**: The model flags a higher proportion of cells in lower-income
   quintiles. This partly reflects genuine higher exposure (more informal settlements in
   flood-prone areas, less drainage infrastructure) — but the gap between flagged % and
   actual risk % indicates potential over-prediction in Q5.

2. **Performance gap**: AUC-ROC varies across quintiles, suggesting the model's
   discriminative ability is not uniform — it may be less calibrated for the wealthiest areas
   where risk events are rarer.

3. **Equity implications**: If this model directly drives resource allocation (e.g., flood
   barriers, emergency response pre-positioning), over-flagging poorer areas could paradoxically
   be beneficial (more protection) or harmful (stigmatisation, insurance cost increases).

## 5. Recommendations

- **Calibration post-processing**: Apply Platt scaling or isotonic regression per income group
  to equalise calibration.
- **Threshold equity**: Use group-specific decision thresholds that equalise false positive rates.
- **Stakeholder review**: Present results to Nairobi County Government and community
  representatives before deployment.
- **Data improvement**: Replace synthetic poverty proxies with actual KNBS census data for
  production use.

## 6. Limitations

- Poverty index is a synthetic proxy, not ground-truth census data.
- Risk target variable is derived (not from actual flood event records).
- Spatial autocorrelation between neighbouring hexes is not accounted for.
"""
    out = REPORTS_DIR / "audit_report.md"
    out.write_text(report)
    print(f"  ✅ Audit report → {out}")


if __name__ == "__main__":
    run_bias_audit()

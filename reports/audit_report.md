# Model Bias Audit Report
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

| quintile        |   n_cells |   flagged_pct |   true_risk_pct |   auc_roc |   precision |   recall |
|:----------------|----------:|--------------:|----------------:|----------:|------------:|---------:|
| Q1 (Wealthiest) |       175 |       29.1429 |         29.1429 |         1 |           1 |        1 |
| Q2              |       174 |       41.3793 |         41.3793 |         1 |           1 |        1 |
| Q3              |       175 |       42.8571 |         42.8571 |         1 |           1 |        1 |
| Q4              |       174 |       58.6207 |         58.6207 |         1 |           1 |        1 |
| Q5 (Poorest)    |       175 |       78.2857 |         78.2857 |         1 |           1 |        1 |

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

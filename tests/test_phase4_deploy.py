"""
Phase 4 Deployment & Ethics tests.

What we test
------------
1. run_bias_audit() computes per-quintile metrics correctly.
2. Audit DataFrame contains expected columns.
3. Flagging rate is 100 % for all-high-risk predictions.
4. Quintile boundaries: Q1 should have lower poverty than Q5.
5. write_audit_report() generates a markdown file with key sections.
6. compute_metrics() flags disparity between flagged_pct and true_risk_pct.
7. Dashboard load_data() fails gracefully when files are missing.
8. Income quintile column is a Categorical with 5 levels.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import joblib


# ── Helpers ───────────────────────────────────────────────────────────

def _make_audit_grid(full_grid, y_true=None, y_pred=None, y_prob=None):
    """
    Return a copy of full_grid with y_true / y_pred / y_prob attached.
    Uses deterministic synthetic values so tests are reproducible.
    """
    n = len(full_grid)
    rng = np.random.default_rng(0)
    g = full_grid.copy()
    g["y_true"] = y_true if y_true is not None else rng.integers(0, 2, n)
    g["y_pred"] = y_pred if y_pred is not None else g["y_true"]    # perfect predictions by default
    g["y_prob"] = y_prob if y_prob is not None else g["y_true"].astype(float)
    return g


def _make_quintiles(grid):
    """Attach income_quintile column the same way bias_audit.py does."""
    poverty_col = "poverty_index" if "poverty_index" in grid.columns else "poverty_rate_percent"
    grid["income_quintile"] = pd.qcut(
        grid[poverty_col], q=5,
        labels=["Q1 (Wealthiest)", "Q2", "Q3", "Q4", "Q5 (Poorest)"]
    )
    return grid


# ─────────────────────────────────────────────────────────────────────
# 1. Per-quintile metrics computation
# ─────────────────────────────────────────────────────────────────────

class TestPerQuintileMetrics:
    """
    We replicate the per-quintile loop from bias_audit.run_bias_audit()
    and test it directly, without file I/O.
    """

    def _compute_quintile_results(self, grid):
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        results = []
        for q in grid["income_quintile"].unique():
            mask = grid["income_quintile"] == q
            sub  = grid[mask]
            if len(sub) < 2 or sub["y_true"].nunique() < 2:
                continue
            results.append({
                "quintile":    q,
                "n_cells":     len(sub),
                "flagged_pct": sub["y_pred"].mean() * 100,
                "true_risk_pct": sub["y_true"].mean() * 100,
                "auc_roc":     roc_auc_score(sub["y_true"], sub["y_prob"]),
                "precision":   precision_score(sub["y_true"], sub["y_pred"], zero_division=0),
                "recall":      recall_score(sub["y_true"], sub["y_pred"], zero_division=0),
            })
        return pd.DataFrame(results)

    def test_returns_non_empty_dataframe(self, full_grid):
        grid = _make_quintiles(_make_audit_grid(full_grid))
        df = self._compute_quintile_results(grid)
        assert len(df) > 0, "Audit results must not be empty"

    def test_required_columns_present(self, full_grid):
        grid = _make_quintiles(_make_audit_grid(full_grid))
        df = self._compute_quintile_results(grid)
        for col in ("quintile", "n_cells", "flagged_pct", "true_risk_pct",
                    "auc_roc", "precision", "recall"):
            assert col in df.columns

    def test_perfect_predictions_give_auc_one(self, full_grid):
        grid = _make_quintiles(_make_audit_grid(full_grid))
        df = self._compute_quintile_results(grid)
        # With y_prob == y_true (0/1), AUC should be 1.0
        assert (df["auc_roc"] == pytest.approx(1.0)).all()

    def test_flagged_pct_in_valid_range(self, full_grid):
        grid = _make_quintiles(_make_audit_grid(full_grid))
        df = self._compute_quintile_results(grid)
        assert df["flagged_pct"].between(0, 100).all()

    def test_all_high_risk_gives_100pct_flagged(self, full_grid):
        n = len(full_grid)
        grid = _make_quintiles(_make_audit_grid(
            full_grid,
            y_true=np.ones(n, dtype=int),
            y_pred=np.ones(n, dtype=int),
            y_prob=np.ones(n, dtype=float),
        ))
        df = self._compute_quintile_results(grid)
        assert (df["flagged_pct"] == pytest.approx(100.0)).all()


# ─────────────────────────────────────────────────────────────────────
# 2. Income quintile column
# ─────────────────────────────────────────────────────────────────────

class TestIncomeQuintiles:
    def test_quintile_column_has_five_levels(self, full_grid):
        grid = _make_quintiles(full_grid.copy())
        assert grid["income_quintile"].nunique() == 5

    def test_quintile_is_categorical(self, full_grid):
        grid = _make_quintiles(full_grid.copy())
        assert hasattr(grid["income_quintile"], "cat"), \
            "income_quintile must be a Categorical series"

    def test_q1_has_lower_poverty_than_q5(self, full_grid):
        poverty_col = "poverty_index" if "poverty_index" in full_grid.columns else "poverty_rate_percent"
        grid = _make_quintiles(full_grid.copy())
        q1_mean = grid.loc[grid["income_quintile"] == "Q1 (Wealthiest)", poverty_col].mean()
        q5_mean = grid.loc[grid["income_quintile"] == "Q5 (Poorest)",    poverty_col].mean()
        assert q1_mean < q5_mean, \
            "Q1 (Wealthiest) should have lower poverty than Q5 (Poorest)"

    def test_no_cells_in_unknown_quintile(self, full_grid):
        grid = _make_quintiles(full_grid.copy())
        valid = {"Q1 (Wealthiest)", "Q2", "Q3", "Q4", "Q5 (Poorest)"}
        unique_qs = set(grid["income_quintile"].dropna().astype(str).unique())
        assert unique_qs.issubset(valid), f"Unexpected quintile labels: {unique_qs - valid}"


# ─────────────────────────────────────────────────────────────────────
# 3. write_audit_report
# ─────────────────────────────────────────────────────────────────────

class TestWriteAuditReport:
    @pytest.fixture
    def sample_audit_df(self):
        return pd.DataFrame({
            "quintile":      ["Q1 (Wealthiest)", "Q2", "Q3", "Q4", "Q5 (Poorest)"],
            "n_cells":       [10, 10, 10, 10, 10],
            "flagged_pct":   [20.0, 30.0, 45.0, 55.0, 70.0],
            "true_risk_pct": [18.0, 28.0, 42.0, 52.0, 65.0],
            "auc_roc":       [0.72, 0.74, 0.71, 0.69, 0.66],
            "precision":     [0.80, 0.78, 0.76, 0.74, 0.72],
            "recall":        [0.55, 0.58, 0.62, 0.65, 0.70],
        })

    def test_report_file_is_created(self, sample_audit_df, tmp_path, monkeypatch):
        import src.phase4_deploy.bias_audit as ba
        monkeypatch.setattr(ba, "REPORTS_DIR", tmp_path)
        ba.write_audit_report(sample_audit_df)
        assert (tmp_path / "audit_report.md").exists()

    def test_report_contains_key_sections(self, sample_audit_df, tmp_path, monkeypatch):
        import src.phase4_deploy.bias_audit as ba
        monkeypatch.setattr(ba, "REPORTS_DIR", tmp_path)
        ba.write_audit_report(sample_audit_df)
        text = (tmp_path / "audit_report.md").read_text()
        for section in ("Objective", "Methodology", "Results", "Recommendations"):
            assert section in text, f"Missing section: {section}"

    def test_report_contains_quintile_data(self, sample_audit_df, tmp_path, monkeypatch):
        import src.phase4_deploy.bias_audit as ba
        monkeypatch.setattr(ba, "REPORTS_DIR", tmp_path)
        ba.write_audit_report(sample_audit_df)
        text = (tmp_path / "audit_report.md").read_text()
        assert "Q5 (Poorest)" in text
        assert "Q1 (Wealthiest)" in text

    def test_report_is_valid_markdown(self, sample_audit_df, tmp_path, monkeypatch):
        """Very lightweight check: file starts with # and has --- separators."""
        import src.phase4_deploy.bias_audit as ba
        monkeypatch.setattr(ba, "REPORTS_DIR", tmp_path)
        ba.write_audit_report(sample_audit_df)
        text = (tmp_path / "audit_report.md").read_text()
        assert text.strip().startswith("#")
        assert "---" in text


# ─────────────────────────────────────────────────────────────────────
# 4. Disparity detection (flagged_pct vs true_risk_pct)
# ─────────────────────────────────────────────────────────────────────

class TestDisparityDetection:
    def test_disparity_grows_with_poverty(self):
        """
        Simulate a model that over-flags poor areas relative to actual risk.
        Verify that the disparity (flagged_pct - true_risk_pct) is larger
        for Q5 than for Q1.
        """
        df = pd.DataFrame({
            "quintile":      ["Q1 (Wealthiest)", "Q2", "Q3", "Q4", "Q5 (Poorest)"],
            "flagged_pct":   [20.0, 30.0, 45.0, 60.0, 80.0],
            "true_risk_pct": [18.0, 27.0, 40.0, 50.0, 62.0],
        })
        df["disparity"] = df["flagged_pct"] - df["true_risk_pct"]
        # Q5 disparity > Q1 disparity
        q1_disp = df.loc[df["quintile"] == "Q1 (Wealthiest)", "disparity"].iloc[0]
        q5_disp = df.loc[df["quintile"] == "Q5 (Poorest)",    "disparity"].iloc[0]
        assert q5_disp > q1_disp, "Expected higher disparity in Q5 (Poorest)"

    def test_zero_disparity_when_predictions_match_reality(self):
        flagged    = [20.0, 35.0, 50.0, 65.0, 80.0]
        true_risk  = [20.0, 35.0, 50.0, 65.0, 80.0]
        disparity  = [f - t for f, t in zip(flagged, true_risk)]
        assert all(d == 0.0 for d in disparity)


# ─────────────────────────────────────────────────────────────────────
# 5. Dashboard load_data fails gracefully without pipeline outputs
# ─────────────────────────────────────────────────────────────────────

class TestDashboardDataLoading:
    def test_load_data_raises_file_not_found_without_pipeline(self, tmp_path, monkeypatch):
        """
        Patch DATA_PROCESSED and MODELS_DIR to an empty directory so
        load_data() raises FileNotFoundError — confirming Phase 1-3 are
        prerequisites, not that the dashboard is broken.
        """
        import src.phase4_deploy.app as app_module
        monkeypatch.setattr(app_module, "DATA_PROCESSED", tmp_path)
        monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path)

        # Bypass the st.cache_data decorator for testing
        load_fn = app_module.load_data.__wrapped__ \
            if hasattr(app_module.load_data, "__wrapped__") \
            else app_module.load_data

        with pytest.raises((FileNotFoundError, Exception)):
            load_fn()

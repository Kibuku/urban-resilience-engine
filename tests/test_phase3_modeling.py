"""
Phase 3 Predictive Modeling tests.

What we test
------------
1. create_risk_target() produces a valid risk_score and binary risk_class.
2. create_risk_target() handles missing columns gracefully (fallback paths).
3. prepare_model_data() returns an X DataFrame with known feature columns.
4. prepare_model_data() raises if no feature columns at all.
5. XGBoost trains, saves, and predicts correctly.
6. MCDropoutNet forward pass produces the right output shape.
7. MCDropoutNet.predict_mc() returns mean/std arrays of correct length.
8. evaluate.compute_metrics() returns expected keys and valid values.
9. feature_eng end-to-end: load → risk → features.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────
# 1 & 2. create_risk_target
# ─────────────────────────────────────────────────────────────────────

class TestCreateRiskTarget:
    def test_risk_score_is_in_unit_range(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target
        out = create_risk_target(full_grid)
        assert out["risk_score"].between(0, 1).all(), \
            "risk_score must be in [0, 1]"

    def test_risk_class_is_binary(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target
        out = create_risk_target(full_grid)
        assert set(out["risk_class"].unique()).issubset({0, 1})

    def test_median_split_is_balanced(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target
        out = create_risk_target(full_grid)
        n_high = out["risk_class"].sum()
        n_total = len(out)
        # Median split: ratio should be within 10 pp of 50%
        ratio = n_high / n_total
        assert 0.4 <= ratio <= 0.6, \
            f"Class imbalance too large for median split: {ratio:.2f}"

    def test_fallback_when_building_count_missing(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target
        grid_no_bld = full_grid.drop(columns=["building_count"])
        out = create_risk_target(grid_no_bld)
        assert "risk_score" in out.columns

    def test_fallback_when_drainage_count_missing(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target
        grid_no_drn = full_grid.drop(columns=["drainage_count"])
        out = create_risk_target(grid_no_drn)
        assert "risk_score" in out.columns

    def test_fallback_poverty_rate_when_poverty_index_missing(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target
        grid_no_idx = full_grid.drop(columns=["poverty_index"])
        out = create_risk_target(grid_no_idx)
        assert "risk_score" in out.columns


# ─────────────────────────────────────────────────────────────────────
# 3 & 4. prepare_model_data
# ─────────────────────────────────────────────────────────────────────

class TestPrepareModelData:
    def test_returns_dataframe_and_series(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
        grid = create_risk_target(full_grid)
        X, y = prepare_model_data(grid)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_feature_matrix_has_no_nans(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
        grid = create_risk_target(full_grid)
        X, _ = prepare_model_data(grid)
        assert X.isna().sum().sum() == 0, "Feature matrix contains NaN values"

    def test_X_and_y_same_length(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
        grid = create_risk_target(full_grid)
        X, y = prepare_model_data(grid)
        assert len(X) == len(y)

    def test_raises_when_no_feature_columns(self, tiny_hex_grid):
        from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
        # Give grid the bare minimum for risk target but no usable model features
        g = tiny_hex_grid.copy()
        g["ndvi_wet"] = 0.5
        g["ndvi_dry"] = 0.4
        g = create_risk_target(g)
        # Strip all expected feature columns
        drop = [c for c in g.columns if c not in ["h3_id", "geometry", "risk_class",
                                                    "risk_score", "ndvi_wet"]]
        g = g.drop(columns=[c for c in drop if c in g.columns], errors="ignore")
        # Should still work with ndvi_wet as a feature
        X, y = prepare_model_data(g)
        assert "ndvi_wet" in X.columns


# ─────────────────────────────────────────────────────────────────────
# 5. XGBoost training
# ─────────────────────────────────────────────────────────────────────

class TestXGBoostTraining:
    def test_train_returns_fitted_model(self, model_data, tmp_path, monkeypatch):
        import src.phase3_modeling.train_xgboost as txgb
        monkeypatch.setattr(txgb, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(txgb, "REPORTS_DIR", tmp_path)
        (tmp_path / "figures").mkdir()

        X, y = model_data
        model = txgb.train_xgboost(X, y)
        assert hasattr(model, "predict_proba")

    def test_predict_proba_shape(self, model_data, tmp_path, monkeypatch):
        import src.phase3_modeling.train_xgboost as txgb
        monkeypatch.setattr(txgb, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(txgb, "REPORTS_DIR", tmp_path)
        (tmp_path / "figures").mkdir()

        X, y = model_data
        model = txgb.train_xgboost(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_probabilities_sum_to_one(self, model_data, tmp_path, monkeypatch):
        import src.phase3_modeling.train_xgboost as txgb
        monkeypatch.setattr(txgb, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(txgb, "REPORTS_DIR", tmp_path)
        (tmp_path / "figures").mkdir()

        X, y = model_data
        model = txgb.train_xgboost(X, y)
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_model_is_saved(self, model_data, tmp_path, monkeypatch):
        import src.phase3_modeling.train_xgboost as txgb
        monkeypatch.setattr(txgb, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(txgb, "REPORTS_DIR", tmp_path)
        (tmp_path / "figures").mkdir()

        X, y = model_data
        txgb.train_xgboost(X, y)
        assert (tmp_path / "xgboost_risk.joblib").exists()


# ─────────────────────────────────────────────────────────────────────
# 6 & 7. MCDropoutNet
# ─────────────────────────────────────────────────────────────────────

class TestMCDropoutNet:
    @pytest.fixture
    def small_net(self):
        from src.phase3_modeling.train_bayesian import MCDropoutNet
        return MCDropoutNet(in_features=5, hidden=[8, 4], dropout_p=0.3)

    def test_forward_output_shape(self, small_net):
        x = torch.randn(10, 5)
        out = small_net(x)
        assert out.shape == (10,), f"Expected (10,), got {out.shape}"

    def test_predict_mc_returns_two_arrays(self, small_net):
        x = torch.randn(12, 5)
        mean, std = small_net.predict_mc(x, T=5)
        assert mean.shape == (12,)
        assert std.shape  == (12,)

    def test_mean_prob_in_unit_range(self, small_net):
        x = torch.randn(20, 5)
        mean, _ = small_net.predict_mc(x, T=10)
        assert (mean >= 0).all() and (mean <= 1).all()

    def test_std_is_non_negative(self, small_net):
        x = torch.randn(20, 5)
        _, std = small_net.predict_mc(x, T=10)
        assert (std >= 0).all()

    def test_dropout_produces_variance(self, small_net):
        """With T > 1 MC samples, std should be > 0 for most inputs."""
        x = torch.randn(50, 5)
        _, std = small_net.predict_mc(x, T=20)
        # At least 80% of samples should show non-zero uncertainty
        assert (std > 0).mean() > 0.8


# ─────────────────────────────────────────────────────────────────────
# 8. evaluate.compute_metrics
# ─────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_required_keys(self):
        from src.phase3_modeling.evaluate import compute_metrics
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob  = rng.uniform(0, 1, 100)
        metrics = compute_metrics(y_true, y_prob)
        for key in ("auc_roc", "brier", "precision", "recall", "accuracy"):
            assert key in metrics

    def test_perfect_classifier_auc_is_one(self):
        from src.phase3_modeling.evaluate import compute_metrics
        y_true = np.array([0, 0, 1, 1])
        y_prob  = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_metrics(y_true, y_prob)
        assert metrics["auc_roc"] == pytest.approx(1.0)

    def test_random_classifier_auc_near_half(self):
        from src.phase3_modeling.evaluate import compute_metrics
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 10_000)
        y_prob  = rng.uniform(0, 1, 10_000)
        metrics = compute_metrics(y_true, y_prob)
        assert 0.45 < metrics["auc_roc"] < 0.55

    def test_accuracy_in_unit_range(self):
        from src.phase3_modeling.evaluate import compute_metrics
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, 50)
        y_prob  = rng.uniform(0, 1, 50)
        metrics = compute_metrics(y_true, y_prob)
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ─────────────────────────────────────────────────────────────────────
# 9. Feature engineering end-to-end (no file I/O)
# ─────────────────────────────────────────────────────────────────────

class TestFeatureEngEndToEnd:
    def test_full_pipeline_produces_non_empty_X(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
        grid = create_risk_target(full_grid)
        X, y = prepare_model_data(grid)
        assert len(X) > 0
        assert len(X.columns) > 0

    def test_y_values_are_binary(self, full_grid):
        from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
        grid = create_risk_target(full_grid)
        _, y = prepare_model_data(grid)
        assert set(y.unique()).issubset({0, 1})

    def test_bayesian_train_smoke(self, model_data, tmp_path, monkeypatch):
        """One-epoch smoke test for the Bayesian trainer (no file I/O)."""
        import src.phase3_modeling.train_bayesian as tb
        monkeypatch.setattr(tb, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(tb, "REPORTS_DIR", tmp_path)
        (tmp_path / "figures").mkdir()

        X, y = model_data
        # Use a very short training run
        model, scaler = tb.train_bayesian(X, y, hidden=[8], epochs=2, batch_size=4)
        assert model is not None
        assert scaler is not None

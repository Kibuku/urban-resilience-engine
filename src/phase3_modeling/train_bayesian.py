"""
Phase 3 — Bayesian Risk Classifier (MC-Dropout Neural Network).

Implements a probabilistic alternative to XGBoost using Monte Carlo Dropout
as a practical approximation to Bayesian inference (Gal & Ghahramani, 2016).

At inference time the dropout layers remain active and the model is run T
times per sample.  The mean of those T forward passes is the predicted
probability; the standard deviation is an epistemic uncertainty estimate.

Usage:
    python src/phase3_modeling/train_bayesian.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Architecture ─────────────────────────────────────────────────────

class MCDropoutNet(nn.Module):
    """
    Feed-forward network with dropout at every hidden layer.
    Dropout is kept active at test time for MC sampling.
    """

    def __init__(self, in_features: int, hidden: list[int] = [64, 32],
                 dropout_p: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout_p)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # binary logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def predict_mc(self, x: torch.Tensor, T: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """
        Run T stochastic forward passes and return (mean_prob, std_prob).

        Parameters
        ----------
        x : Input tensor (N, F).
        T : Number of MC samples.

        Returns
        -------
        mean_prob : Shape (N,) — predicted probability (risk score).
        std_prob  : Shape (N,) — epistemic uncertainty per sample.
        """
        self.train()           # keep dropout active
        with torch.no_grad():
            samples = torch.stack(
                [torch.sigmoid(self.forward(x)) for _ in range(T)], dim=0
            )                  # (T, N)
        return samples.mean(0).numpy(), samples.std(0).numpy()


# ── Training ─────────────────────────────────────────────────────────

def train_bayesian(
    X: pd.DataFrame,
    y: pd.Series,
    hidden: list[int] = [64, 32],
    dropout_p: float = 0.3,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> tuple[MCDropoutNet, StandardScaler]:
    """Train the MC-Dropout network and return (model, scaler)."""
    print("⏳ Phase 3 (Bayesian): Training MC-Dropout network...")

    # Scale features
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X.values).astype(np.float32)
    y_np = y.values.astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sc, y_np, test_size=0.2, stratify=y_np, random_state=SEED
    )

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )

    model = MCDropoutNet(X.shape[1], hidden=hidden, dropout_p=dropout_p)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    val_tensor = torch.tensor(X_val)
    best_auc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            mean_prob, _ = model.predict_mc(val_tensor, T=30)
            auc = roc_auc_score(y_val, mean_prob)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(tr_loader):.4f}  "
                  f"val_AUC={auc:.4f}")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  ✅ Best val AUC: {best_auc:.4f}")
    return model, scaler


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_bayesian(
    model: MCDropoutNet,
    scaler: StandardScaler,
    X: pd.DataFrame,
    y: pd.Series,
    T: int = 100,
) -> pd.DataFrame:
    """
    Evaluate with MC sampling; return per-sample DataFrame with
    predicted probability, uncertainty, and true label.
    """
    X_sc = torch.tensor(scaler.transform(X.values).astype(np.float32))
    mean_prob, std_prob = model.predict_mc(X_sc, T=T)
    y_pred = (mean_prob >= 0.5).astype(int)

    auc   = roc_auc_score(y, mean_prob)
    brier = brier_score_loss(y, mean_prob)
    print(f"\n📊 Bayesian model evaluation (T={T} MC samples):")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  Mean uncertainty (std): {std_prob.mean():.4f}")

    results = pd.DataFrame({
        "y_true":      y.values,
        "y_pred":      y_pred,
        "prob_mean":   mean_prob,
        "prob_std":    std_prob,
    })

    # Uncertainty calibration plot
    _plot_uncertainty(results, brier, auc)
    return results


def _plot_uncertainty(results: pd.DataFrame, brier: float, auc: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(results["prob_mean"], results["prob_std"],
                    c=results["y_true"], cmap="RdYlGn_r", alpha=0.6, s=20)
    axes[0].set_xlabel("Predicted probability (mean)")
    axes[0].set_ylabel("Epistemic uncertainty (std)")
    axes[0].set_title("Uncertainty vs Predicted Risk\n(coloured by true label)")

    axes[1].hist(results.loc[results["y_true"] == 0, "prob_mean"],
                 bins=30, alpha=0.6, label="Low Risk (true)", color="green")
    axes[1].hist(results.loc[results["y_true"] == 1, "prob_mean"],
                 bins=30, alpha=0.6, label="High Risk (true)", color="red")
    axes[1].axvline(0.5, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Probability Histogram  AUC={auc:.3f}  Brier={brier:.3f}")
    axes[1].legend()

    fig.tight_layout()
    out = REPORTS_DIR / "figures" / "bayesian_uncertainty.png"
    fig.savefig(out, dpi=150)
    print(f"  ✅ Uncertainty plot → {out}")
    plt.close()


# ── Persistence ───────────────────────────────────────────────────────

def save_bayesian_model(model: MCDropoutNet, scaler: StandardScaler) -> None:
    torch.save(model.state_dict(), MODELS_DIR / "bayesian_net.pt")
    joblib.dump(scaler, MODELS_DIR / "bayesian_scaler.joblib")
    # Also save architecture kwargs for easy reload
    joblib.dump(
        {"in_features": next(model.net[0].parameters()).shape[1],
         "hidden": [l.out_features for l in model.net if isinstance(l, nn.Linear)][:-1],
         "dropout_p": model.net[2].p if len(model.net) > 2 else 0.3},
        MODELS_DIR / "bayesian_arch.joblib"
    )
    print(f"  ✅ Bayesian model → {MODELS_DIR / 'bayesian_net.pt'}")


def load_bayesian_model(in_features: int) -> tuple[MCDropoutNet, StandardScaler]:
    arch   = joblib.load(MODELS_DIR / "bayesian_arch.joblib")
    model  = MCDropoutNet(**arch)
    model.load_state_dict(torch.load(MODELS_DIR / "bayesian_net.pt", weights_only=True))
    scaler = joblib.load(MODELS_DIR / "bayesian_scaler.joblib")
    return model, scaler


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X = pd.read_parquet(DATA_PROCESSED / "X_features.parquet")
    y = pd.read_parquet(DATA_PROCESSED / "y_target.parquet")["risk_class"]

    model, scaler = train_bayesian(X, y)
    save_bayesian_model(model, scaler)
    results = evaluate_bayesian(model, scaler, X, y)

    print("\n🏁 Phase 3 (Bayesian) complete.")

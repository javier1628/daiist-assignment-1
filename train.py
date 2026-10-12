"""Train every model once and save the artefacts the dashboard loads.

This is the ONLY script that fits anything. It:
  1. loads the data and makes THE train/test split (src.data)
  2. builds features via the owner's build_features (src.features)
  3. fits two baselines, Ridge, the manual PyTorch loop, and an nn.Linear model
  4. writes everything under models/ as PLAN.md section 5 lists

Run from the repo root:  uv run python main.py train

Hyperparameters (alpha / lr / weight_decay) are chosen on a validation split carved
out of TRAIN; the test set is scored exactly once, at the end. The single train/test
split in src.data stays the authority for the final comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src import data
from src.features import build_features, code_prefix
from src.train_manual import fit_manual, predict_manual

# Reproducibility: the data split already uses random_state=42; pin torch too so the
# manual loop and the nn.Linear model give the same numbers on every run.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# Hyperparameter grids. PLAN.md limits tuning to these and nothing else.
RIDGE_ALPHAS = [0.1, 1.0, 10.0]
MANUAL_LRS = [0.1, 0.01, 0.001]
MANUAL_EPOCHS = 2000
# nn.Linear + Adam: lr is fixed (Adam is forgiving); weight_decay is matched to the
# chosen Ridge alpha below so all three models land on ~the same coefficients.
TORCH_LR = 0.05
TORCH_EPOCHS = 3000


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Errors in log space (what we optimise) plus a business-readable % error in DOP.

    - rmse_log / mae_log: on log(Costo), the quantity every model is trained on.
    - mdape_pct: median absolute percentage error back in DOP, so the number means
      something to the sales staff (PLAN.md section 2).
    """
    err = y_pred_log - y_true_log
    rmse_log = float(np.sqrt(np.mean(err ** 2)))
    mae_log = float(np.mean(np.abs(err)))
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    mdape_pct = float(np.median(np.abs(y_pred - y_true) / y_true) * 100.0)
    return {"rmse_log": rmse_log, "mae_log": mae_log, "mdape_pct": mdape_pct}


# ----------------------------------------------------------------------------- torch model
def train_torch_module(
    X: np.ndarray, y: np.ndarray, weight_decay: float, lr: float, epochs: int
) -> tuple[nn.Linear, list[float]]:
    """A single nn.Linear trained with Adam + MSE. weight_decay is L2 on the WEIGHT
    only (not the bias), to match Ridge's unpenalised intercept and the manual loop."""
    torch.manual_seed(SEED)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model = nn.Linear(X.shape[1], 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [
            {"params": [model.weight], "weight_decay": weight_decay},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    loss_curve: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))
    return model, loss_curve


def torch_predict(model: nn.Linear, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


# ----------------------------------------------------------------------------- main
def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Data + THE split (identical for every model, defined once in src.data).
    df = data.load_raw()
    train_df, test_df = data.split(df)
    y_train = data.make_target(train_df).to_numpy(dtype=float)
    y_test = data.make_target(test_df).to_numpy(dtype=float)

    # 2. Features. fit is learned on TRAIN and reused (learns nothing) on TEST.
    X_train_df, fit = build_features(train_df, fit=None)
    X_test_df, _ = build_features(test_df, fit=fit)
    feature_names = list(fit["columns"])
    # build_features promises stable column order; make the arrays follow it explicitly.
    X_train = X_train_df[feature_names].to_numpy(dtype=float)
    X_test = X_test_df[feature_names].to_numpy(dtype=float)
    n_train = X_train.shape[0]
    print(f"features: {X_train.shape[1]}  |  train {n_train}  test {X_test.shape[0]}")

    # Validation split carved out of TRAIN, used only to pick hyperparameters.
    Xt, Xv, yt, yv = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED
    )

    metrics: dict = {
        "n_train": n_train,
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }

    # 3a. Baseline A: predict the train mean of log(Costo).
    mean_log = float(np.mean(y_train))
    baseline_a_pred = np.full_like(y_test, mean_log)
    metrics["baseline_mean"] = {"value": mean_log, **regression_metrics(y_test, baseline_a_pred)}

    # 3b. Baseline B: train mean of log(Costo) per Código prefix, global mean as fallback.
    train_prefix = train_df["Código"].map(code_prefix)
    test_prefix = test_df["Código"].map(code_prefix)
    prefix_mean = pd.Series(y_train, index=train_prefix.values).groupby(level=0).mean()
    baseline_b_pred = test_prefix.map(prefix_mean).fillna(mean_log).to_numpy(dtype=float)
    metrics["baseline_prefix_mean"] = regression_metrics(y_test, baseline_b_pred)

    # 3c. Ridge: choose alpha on the validation split, refit on full train, score test once.
    ridge_candidates = {}
    for alpha in RIDGE_ALPHAS:
        val_model = Ridge(alpha=alpha).fit(Xt, yt)
        ridge_candidates[str(alpha)] = regression_metrics(yv, val_model.predict(Xv))
    best_alpha = min(RIDGE_ALPHAS, key=lambda a: ridge_candidates[str(a)]["rmse_log"])
    ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    metrics["ridge"] = {
        "chosen_alpha": best_alpha,
        "val_candidates": ridge_candidates,
        "test": regression_metrics(y_test, ridge_pred),
    }

    # Match the manual loop and Adam to Ridge's L2. sklearn minimises
    #   sum_i (y - Xw)^2 + alpha * ||w||^2
    # while the manual loss and nn loss use the MEAN squared error:
    #   mean_i (y - Xw)^2 + l2 * ||w||^2.
    # Multiplying the mean form by n shows the equivalent penalty is l2 = alpha / n.
    # Using it lets all three converge to ~the same coefficients (PLAN.md section 5).
    l2 = best_alpha / n_train
    l2_val = best_alpha / Xt.shape[0]  # same idea, sized to the smaller val-search fit

    # 3d. Manual PyTorch loop: choose lr on validation, refit on full train.
    manual_candidates = {}
    for lr in MANUAL_LRS:
        params_val = fit_manual(Xt, yt, lr=lr, l2=l2_val, epochs=MANUAL_EPOCHS)
        manual_candidates[str(lr)] = regression_metrics(yv, predict_manual(params_val, Xv))
    best_lr = min(MANUAL_LRS, key=lambda r: manual_candidates[str(r)]["rmse_log"])
    manual_params = fit_manual(X_train, y_train, lr=best_lr, l2=l2, epochs=MANUAL_EPOCHS)
    manual_pred = predict_manual(manual_params, X_test)
    metrics["manual"] = {
        "chosen_lr": best_lr,
        "l2": l2,
        "epochs": MANUAL_EPOCHS,
        "val_candidates": manual_candidates,
        "test": regression_metrics(y_test, manual_pred),
    }

    # 3e. nn.Linear + Adam, weight_decay matched to Ridge (weight only).
    torch_model, torch_loss_curve = train_torch_module(
        X_train, y_train, weight_decay=l2, lr=TORCH_LR, epochs=TORCH_EPOCHS
    )
    torch_pred = torch_predict(torch_model, X_test)
    metrics["torch_module"] = {
        "lr": TORCH_LR,
        "weight_decay": l2,
        "epochs": TORCH_EPOCHS,
        "test": regression_metrics(y_test, torch_pred),
    }

    # 4. Save everything PLAN.md section 5 lists.
    # NOTE: build_features stores the StandardScaler inside the fit dict. The contract
    # only guarantees fit["columns"], so if the scaler key differs, change it here.
    scaler = fit.get("scaler")
    if scaler is None:
        raise KeyError(
            "fit['scaler'] not found — build_features must put the trained StandardScaler "
            "under fit['scaler'] (see train.py save step)."
        )
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    # The whole fit dict (frequency tables, top-N lists, scaler, size median, column
    # order) — app.py needs it to featurize a single SKU typed into the dashboard.
    joblib.dump(fit, MODELS_DIR / "feature_fit.joblib")
    joblib.dump(ridge, MODELS_DIR / "ridge.joblib")
    torch.save({"w": manual_params["w"], "b": manual_params["b"]}, MODELS_DIR / "manual.pt")
    torch.save(torch_model.state_dict(), MODELS_DIR / "torch_module.pt")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Loss curves live in their own file: thousands of floats per model would otherwise
    # bury the handful of numbers metrics.json is meant to be read for.
    loss_curves = {
        "manual": [float(v) for v in manual_params["loss_curve"]],
        "torch_module": torch_loss_curve,
    }
    (MODELS_DIR / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2))

    # test_predictions.csv: one row per test SKU, actual + every model, log and DOP space,
    # for the dashboard's predicted-vs-actual scatter and residual histogram (PLAN.md section 6).
    preds = pd.DataFrame(
        {
            "Código": test_df["Código"].to_numpy(),
            "Descripción": test_df["Descripción"].to_numpy(),
            "Costo": test_df["Costo"].to_numpy(),
            "y_log_actual": y_test,
            "pred_baseline_mean_log": baseline_a_pred,
            "pred_baseline_prefix_log": baseline_b_pred,
            "pred_ridge_log": ridge_pred,
            "pred_manual_log": manual_pred,
            "pred_torch_log": torch_pred,
        }
    )
    for col in ["ridge", "manual", "torch"]:
        preds[f"pred_{col}_dop"] = np.exp(preds[f"pred_{col}_log"])
    preds.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    # 5. Console summary so the owner can eyeball convergence.
    print("\n=== test-set metrics (log space RMSE / MAE, DOP MdAPE%) ===")
    rows = [
        ("baseline mean", metrics["baseline_mean"]),
        ("baseline prefix", metrics["baseline_prefix_mean"]),
        (f"ridge (a={best_alpha})", metrics["ridge"]["test"]),
        (f"manual (lr={best_lr})", metrics["manual"]["test"]),
        ("nn.Linear+Adam", metrics["torch_module"]["test"]),
    ]
    print(f"{'model':<22}{'RMSE':>8}{'MAE':>8}{'MdAPE%':>9}")
    for name, m in rows:
        print(f"{name:<22}{m['rmse_log']:>8.4f}{m['mae_log']:>8.4f}{m['mdape_pct']:>9.1f}")
    print(f"\nsaved artefacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()

"""Gradio dashboard for the SKU cost estimator.

Loads the artefacts train.py wrote under models/ and serves three tabs:
  1. Data      — what the catalogue and the target look like
  2. Models    — how the three models score and agree
  3. Estimator — type a SKU, get three cost estimates with a ±1 RMSE band

This script NEVER trains. It loads from models/ and fails loudly if a file is missing.
The only raw data it reads is the CSV, and only to draw the Data tab's charts.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend: we hand figures to Gradio, never display them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import gradio as gr

from src import data
from src.features import build_features, family, material, size_cm

MODELS_DIR = Path(__file__).resolve().parent / "models"


# ----------------------------------------------------------------------------- load artefacts
def _require(path: Path) -> Path:
    """Fail loudly: the dashboard loads trained models, it never retrains them."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run `uv run python main.py train` first.")
    return path


feature_fit = joblib.load(_require(MODELS_DIR / "feature_fit.joblib"))
ridge = joblib.load(_require(MODELS_DIR / "ridge.joblib"))
manual = torch.load(_require(MODELS_DIR / "manual.pt"))  # {"w": tensor, "b": tensor}
feature_names = json.loads(_require(MODELS_DIR / "feature_names.json").read_text())
metrics = json.loads(_require(MODELS_DIR / "metrics.json").read_text())
test_preds = pd.read_csv(_require(MODELS_DIR / "test_predictions.csv"))

# Rebuild the nn.Linear and load its saved weights.
torch_model = nn.Linear(len(feature_names), 1)
torch_model.load_state_dict(torch.load(_require(MODELS_DIR / "torch_module.pt")))
torch_model.eval()

# Per-model test RMSE (log space) → drives the ±1 RMSE band in the estimator.
RMSE = {
    "Ridge": metrics["ridge"]["test"]["rmse_log"],
    "Manual": metrics["manual"]["test"]["rmse_log"],
    "nn.Linear": metrics["torch_module"]["test"]["rmse_log"],
}

# Raw data, only for the Data tab's charts (reading, not retraining).
raw_df = data.load_raw()


# ----------------------------------------------------------------------------- prediction
def _coefficients() -> dict[str, np.ndarray]:
    """The learned weight vector of each model, aligned to feature_names."""
    return {
        "Ridge": ridge.coef_,
        "Manual": manual["w"].numpy(),
        "nn.Linear": torch_model.weight.detach().numpy().ravel(),
    }


def _intercepts() -> dict[str, float]:
    return {
        "Ridge": float(ridge.intercept_),
        "Manual": float(manual["b"]),
        "nn.Linear": float(torch_model.bias.detach()),
    }


def _predict_log(X: np.ndarray) -> dict[str, np.ndarray]:
    """Each model's prediction of log(Costo) for already-scaled feature rows."""
    coefs, biases = _coefficients(), _intercepts()
    out = {name: X @ coefs[name] + biases[name] for name in ("Ridge", "Manual")}
    with torch.no_grad():
        out["nn.Linear"] = torch_model(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    return out


# ----------------------------------------------------------------------------- Tab 1: Data
def plot_cost_histogram():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw_df["Costo"], bins=np.logspace(np.log10(raw_df["Costo"].min()),
                                              np.log10(raw_df["Costo"].max()), 40))
    ax.set_xscale("log")
    ax.set_xlabel("Costo (DOP, log scale)")
    ax.set_ylabel("count")
    ax.set_title("Cost distribution")
    fig.tight_layout()
    return fig


def plot_target_distribution():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log(raw_df["Costo"]), bins=40)
    ax.set_xlabel("log(Costo) — the modelling target")
    ax.set_ylabel("count")
    ax.set_title("Target distribution")
    fig.tight_layout()
    return fig


def _bar(series_counts, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    series_counts.plot.bar(ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_family_bar():
    return _bar(raw_df["Descripción"].astype(str).map(family).value_counts(),
                "Product family", "family")


def plot_material_bar():
    return _bar(raw_df["Descripción"].astype(str).map(material).value_counts(),
                "Material", "material")


def plot_size_vs_cost():
    size = raw_df["Descripción"].astype(str).map(size_cm)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(size, np.log(raw_df["Costo"]), s=8, alpha=0.3)
    ax.set_xlabel("parsed size (cm)")
    ax.set_ylabel("log(Costo)")
    ax.set_title("Size vs log cost (where a size was parsed)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Tab 2: Models
def metrics_table() -> pd.DataFrame:
    def row(name, m):
        return {"model": name, "RMSE (log)": round(m["rmse_log"], 3),
                "MAE (log)": round(m["mae_log"], 3), "Median % err (DOP)": round(m["mdape_pct"], 1)}
    rows = [
        row("Baseline: mean", metrics["baseline_mean"]),
        row("Baseline: prefix mean", metrics["baseline_prefix_mean"]),
        row("Ridge", metrics["ridge"]["test"]),
        row("Manual", metrics["manual"]["test"]),
        row("nn.Linear", metrics["torch_module"]["test"]),
    ]
    return pd.DataFrame(rows)


def plot_pred_vs_actual(model_name: str):
    actual = test_preds["y_log_actual"]
    pred = test_preds[f"pred_{model_name.lower().replace('nn.linear', 'torch')}_log"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, pred, s=8, alpha=0.3)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)  # perfect-prediction line
    ax.set_xlabel("actual log(Costo)")
    ax.set_ylabel("predicted log(Costo)")
    ax.set_title(f"Predicted vs actual — {model_name}")
    fig.tight_layout()
    return fig


def plot_residuals(model_name: str):
    pred = test_preds[f"pred_{model_name.lower().replace('nn.linear', 'torch')}_log"]
    resid = pred - test_preds["y_log_actual"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=40)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("residual (predicted − actual, log space)")
    ax.set_ylabel("count")
    ax.set_title(f"Residuals — {model_name}")
    fig.tight_layout()
    return fig


def plot_coefficients():
    coefs = _coefficients()
    order = np.argsort(np.abs(coefs["Ridge"]))[::-1]  # most influential first
    names = [feature_names[i] for i in order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 11))
    for offset, name in zip((-0.25, 0.0, 0.25), ("Ridge", "Manual", "nn.Linear")):
        ax.barh(y + offset, coefs[name][order], height=0.25, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient (standardised features)")
    ax.set_title("Coefficients — the three models agree")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Tab 3: Estimator
def estimate(desc, prefix, dept, supplier, area, empaque) -> pd.DataFrame:
    # Turn the chosen prefix back into a Código whose code_prefix() == prefix, because
    # build_features derives the prefix from Código, not from a prefix field.
    if prefix == "NUM":
        codigo = "99999"           # purely numeric → code_prefix returns "NUM"
    elif prefix == "OTHER":
        codigo = "ZZZZ9999"        # a prefix that won't be in the train top-N → OTHER
    else:
        codigo = prefix

    # One raw row with the SAME dtypes as the training data, so astype(str) inside
    # build_features produces frequency-table keys that match ("218", not "218.0").
    row = pd.DataFrame([{
        "Descripción": desc or "",
        "Código": codigo,
        "Dpto (ID)": int(dept),
        "Suplidor (ID)": int(supplier),
        "Área (ID)": int(area),
        "Empaque": int(empaque),
    }])

    X, _ = build_features(row, fit=feature_fit)   # reuse fit: learns nothing
    preds_log = _predict_log(X.to_numpy())

    rows = []
    for name in ("Ridge", "Manual", "nn.Linear"):
        p = float(preds_log[name][0])
        rmse = RMSE[name]
        rows.append({
            "model": name,
            "estimate (DOP)": round(float(np.exp(p))),
            "low (−1 RMSE)": round(float(np.exp(p - rmse))),
            "high (+1 RMSE)": round(float(np.exp(p + rmse))),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- UI
MODEL_CHOICES = ["Ridge", "Manual", "nn.Linear"]
PREFIX_CHOICES = feature_fit["prefix_levels"]        # top-N code prefixes + OTHER
DEPT_CHOICES = feature_fit["top_depts"] + ["(other)"]  # top-8 departments + a catch-all

with gr.Blocks(title="Display Internacional — SKU cost estimator") as demo:
    gr.Markdown("# Display Internacional — SKU cost estimator\n"
                "Estimates of landed **cost** (log model, DOP). A guardrail, not a price oracle.")

    with gr.Tab("Data"):
        with gr.Row():
            gr.Plot(plot_cost_histogram())
            gr.Plot(plot_target_distribution())
        with gr.Row():
            gr.Plot(plot_family_bar())
            gr.Plot(plot_material_bar())
        gr.Plot(plot_size_vs_cost())

    with gr.Tab("Models"):
        gr.Dataframe(metrics_table(), label="Test-set metrics")
        model_pick = gr.Dropdown(MODEL_CHOICES, value="Ridge", label="Model for the plots below")
        with gr.Row():
            pva = gr.Plot(plot_pred_vs_actual("Ridge"))
            resid = gr.Plot(plot_residuals("Ridge"))
        model_pick.change(plot_pred_vs_actual, model_pick, pva)
        model_pick.change(plot_residuals, model_pick, resid)
        gr.Plot(plot_coefficients())

    with gr.Tab("Estimator"):
        gr.Markdown("Type what you know about the SKU. Unseen IDs are handled (encoded as 0).")
        with gr.Row():
            in_desc = gr.Textbox(label="Descripción", placeholder="e.g. GANCHO METAL 4 PLG")
            in_prefix = gr.Dropdown(PREFIX_CHOICES, value=PREFIX_CHOICES[0], label="Código prefix")
        with gr.Row():
            in_dept = gr.Dropdown(DEPT_CHOICES, value=DEPT_CHOICES[0], label="Dpto (ID)")
            in_supplier = gr.Number(value=0, precision=0, label="Suplidor (ID)")
            in_area = gr.Number(value=0, precision=0, label="Área (ID)")
            in_empaque = gr.Number(value=1, precision=0, label="Empaque (units/pack)")
        go = gr.Button("Estimate cost", variant="primary")
        out = gr.Dataframe(label="Estimated cost with ±1 RMSE band (DOP)")

        def _estimate(desc, prefix, dept, supplier, area, empaque):
            dept_val = -1 if dept == "(other)" else dept  # catch-all → unseen dept id
            return estimate(desc, prefix, dept_val, supplier, area, empaque)

        go.click(_estimate, [in_desc, in_prefix, in_dept, in_supplier, in_area, in_empaque], out)


if __name__ == "__main__":
    demo.launch()

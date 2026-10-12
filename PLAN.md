# Plan

## 1. Business problem
Display Internacional SRL (Santo Domingo) distributes retail fixtures: hooks, hangers, shelving,
checkouts, showcases, mannequins. Sales staff quote customers daily, often on items whose
landed cost is not yet known (new SKU, supplier quote pending, product not bought in years).
A model that estimates landed cost from a product's attributes lets them (a) price a quote
before the supplier answers and (b) sanity-check a supplier quote that looks off.

Decision supported: "What should this SKU roughly cost us?" — used as a guardrail, not as the
price itself. Tolerance: being within ~±40% is useful; the bands the dashboard shows matter more
than the point estimate.

## 2. Target
`y = log(Costo)`. Cost spans 0.08 → 3.2M DOP (median ~950), so raw cost makes a linear model
chase a handful of giant items. Log turns multiplicative markup/size effects into additive ones.
Report errors both in log space (RMSE, MAE) and back-transformed as a median absolute
percentage error so the number means something to the business.

## 3. Features (owner writes `src/features.py`)
From `Descripción` (free text, Spanish/English mix):
- product family: keyword match → GANCHO, PERCHA, ESTANTERIA/TRAMERIA, CHECK-OUT, VITRINA,
  MANIQUI, EXHIBIDOR, PORTA PRECIO, CARRO/CARRETILLA, CERRADURA, OTHER
- material: METAL/NIQUEL/ZINC/CROMO, MADERA, ACRILICO, PLASTICO, CRISTAL/VIDRIO, OTHER
- size: first number before `Plg` or `"` (inches) and first number before `mm`/`MTS`
  (convert to a single `size_cm` where possible); flag `has_dims` for `AxBxC` patterns
- `n_words`, `has_color` (BLANCO/NEGRO/GRIS/AZUL/ROJO/…)
From `Código`: alphabetic prefix, top 10 + OTHER (813 codes are purely numeric → `NUM`)
Categoricals: `Dpto (ID)`, `Suplidor (ID)`, `Área (ID)` → frequency encoding (count in train),
plus one-hot for the top 8 departments. Document why not full one-hot (74 suppliers / 2k rows).
Numeric: `Empaque`, `log1p(Empaque)`.
Standardise everything with `StandardScaler` fit on train.

## 4. Split
Random 80/20, `random_state=42`, stratified by product family bins is optional. No time split is
possible (snapshot, no dates) — state this as a limitation.

## 5. Models (all on identical X_train/X_test)
- Baseline A: predict train mean of log(Costo). Baseline B: train mean per code prefix.
- `sklearn.linear_model.Ridge(alpha)` — try alpha ∈ {0.1, 1, 10}
- Manual PyTorch: `w`, `b` tensors with `requires_grad`, MSE + L2, explicit `w -= lr * w.grad`,
  `zero_()` grads, full-batch, ~2000 epochs. lr ∈ {0.1, 0.01, 0.001}. Log loss curve.
- `nn.Linear(n_features, 1)` + `nn.MSELoss` + `torch.optim.Adam(weight_decay=…)`.
Expect all three to converge to ~the same coefficients; if they don't, that is the interesting
finding to explain (lr too high, not enough epochs, L2 parameterised differently).
Save: `models/scaler.joblib`, `models/feature_fit.joblib` (the whole fit dict so `app.py`
can featurize new input), `models/ridge.joblib`, `models/manual.pt` (w, b),
`models/torch_module.pt` (state_dict), `models/feature_names.json`, `models/metrics.json`,
`models/test_predictions.csv`.

## 6. Dashboard (`app.py`, Gradio)
Tab 1 Data: histogram of Costo (log x-axis), target distribution, bar charts of product family
and material, scatter of size vs log cost.
Tab 2 Models: metrics table (baselines + 3 models), predicted-vs-actual scatter with model
dropdown, residual histogram, coefficient bar chart comparing the three models.
Tab 3 Estimator: free-text description + prefix + dept + supplier + pack size → three cost
estimates side by side with a ±1 RMSE band in DOP.
Startup: load from `models/` only. No training code imported.

## 7. REPORT.md sections
Dataset · Business framing · Target & leakage decisions · Features (what, why, what failed) ·
Split · Three-method comparison table · Baselines · What didn't work · Limitations · How to run.

## 8. Known limitations to state honestly
Snapshot not history; supplier/dept are opaque IDs; descriptions are inconsistent (mixed
languages, abbreviations); 2k rows limits how many categorical levels we can afford; cost is
historical average cost, not current landed cost; currency assumed DOP; some costs are clearly
stale (e.g. items with cost < 1 DOP).

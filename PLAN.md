# Plan

## 1. Business problem
Display Internacional SRL (Santo Domingo) distributes retail fixtures: hooks, hangers, shelving,
checkouts, showcases, mannequins. The July 2025 snapshot has ~2,100 SKUs, ~1,300 of them with
zero or negative stock. Some of those are simply sold out and will be reordered; many are
**dead stock**: items nobody intends to restock, still cluttering the catalogue, price lists and
the warehouse system. Reviewing 1,300 items one by one is not going to happen.

Decision supported: "Which SKUs should the family review for discontinuation?" The model scores
every SKU with a probability of being dead and ranks them, so the review covers the ~200 most
likely candidates instead of the whole catalogue. It is a triage tool: it shortlists, a person
decides.

Cost assumptions (the owner's, marked as assumptions, constants in `src/data.py`):
- **False positive** — flagging a live item, which gets discontinued by mistake and has to be
  re-listed or is lost as a sale: `FP_COST` DOP.
- **False negative** — leaving a dead item listed: `FN_COST` DOP (catalogue/admin/storage drag).
These two numbers set the decision threshold (§5) and drive the live cost figure in the dashboard.

## 2. Target
Binary: `dead = (Stock <= 0) & (Punto reorden == 0)`, on `Tipo == 'Producto'` rows with
`Costo > 0` (2,067 rows). Reading: nothing on hand AND no reorder point set, i.e. the system
has no intention of restocking. Zero stock alone is not enough (sold-out items get reordered);
a zero reorder point alone is not enough (plenty of stocked items have none). 1,253 rows
(60.6%) are dead, so "dead" is the majority class and the majority baseline already scores
~61% accuracy. Ranking quality (ROC-AUC) and cost at the threshold are what matter.

Leakage: the label is built from `Stock` and `Punto reorden`, and `Estado stock`,
`Valor inventario`, `Cant. reorden`, `Alertas` are functions of the same columns, so all six are
banned. Price columns (`Precio detalle`, `Precio mayor 1/2/3`, `Precio mínimo`, `Margen %`) are
markup × cost and carry nothing beyond `Costo`, so they stay excluded too. `Costo` itself is now a
legitimate feature: expensive items turn over slowly and are plausibly more often dead.

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
Numeric: `log1p(Empaque)` only (raw `Empaque` dropped as a near-duplicate). `log(Costo)` was
tried and dropped: test AUC 0.7206 with it, 0.7217 without.
Standardise everything with `StandardScaler` fit on train.

## 4. Split
Random 80/20, `random_state=42`. No time split is possible (snapshot, no dates) — state this as
a limitation. A validation slice (20% of train) is carved out only to pick hyperparameters and
the threshold; the test set is scored once at the end.

## 5. Models (all on identical X_train/X_test)
- Baseline A: majority class (always "dead"). Baseline B: dead if `Costo` > train median.
- `sklearn.linear_model.LogisticRegression(C)` — try C ∈ {0.01, 0.1, 1, 10}, pick by val AUC.
- Manual PyTorch: `w`, `b` tensors with `requires_grad`, loss = mean BCE on logits
  (`binary_cross_entropy_with_logits`, numerically stable) + l2·‖w‖², explicit `w -= lr * w.grad`,
  `zero_()` grads, full-batch, ~2000 epochs. lr ∈ {0.1, 0.01, 0.001} by val AUC. Log loss curve.
- `nn.Linear(n_features, 1)` + `nn.BCEWithLogitsLoss` + `torch.optim.Adam(weight_decay=…)`.
Regularisation matched across the three: sklearn minimises `C·Σ BCE + ½‖w‖²`; dividing by C·n
gives the mean-loss form `mean BCE + l2·‖w‖²` with **`l2 = 1 / (2·C·n)`**.
Threshold: the one that minimises `FP_COST·FP + FN_COST·FN` on the validation slice (scanned
over the sklearn model's probabilities). Test metrics are reported at 0.5 and at that threshold.
Metrics: accuracy, precision, recall, F1, ROC-AUC, expected cost (DOP) at the threshold.
Expect all three to converge to ~the same coefficients; if they don't, that is the interesting
finding to explain (lr too high, not enough epochs, L2 parameterised differently).
Save: `models/scaler.joblib`, `models/feature_fit.joblib` (the whole fit dict so `app.py`
can featurize new input), `models/logreg.joblib`, `models/manual.pt` (w, b),
`models/torch_module.pt` (state_dict), `models/feature_names.json`, `models/metrics.json`
(no curves), `models/loss_curves.json`, `models/test_predictions.csv` (y_true + one
probability column per model).

## 6. Dashboard (`app.py`, Gradio, matplotlib)
Tab 1 Data: class balance bar; log-cost histogram split by class; family and material bars
split by class rate; supplier-frequency vs dead-rate scatter.
Tab 2 Models: metrics table (both baselines + 3 models, at 0.5 and at the chosen threshold);
ROC curves for all three on one plot; predicted-probability histograms split by true class with a
model dropdown; coefficient bar chart comparing the three models.
Tab 3 Threshold: slider 0–1 + model dropdown → confusion matrix as a 2×2 heatmap and the
expected business cost in DOP computed live from `FP_COST` / `FN_COST`, with a note showing the
cost-minimising threshold.
Tab 4 Scorer: description, code prefix, dept, supplier, area, pack size, cost → three dead-stock
probabilities side by side.
Startup: load from `models/` only. No training code imported.

## 7. REPORT.md sections
Dataset · Business framing · Target & leakage decisions · Features (what, why, what failed) ·
Split · Three-method comparison table · Baselines · Threshold & cost · What didn't work ·
Limitations · How to run.

## 8. Known limitations to state honestly
Snapshot not history (no sales velocity, no "last sold" date — the single most useful signal
for dead stock is missing); the label is a proxy built from reorder settings, which may be
unset for reasons other than "dead"; supplier/dept are opaque IDs; descriptions are
inconsistent; 2k rows limits how many categorical levels we can afford; FP/FN costs are
assumptions, not measured; currency assumed DOP.

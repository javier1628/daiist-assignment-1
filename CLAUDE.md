# Display Internacional — dead-stock classifier (Assignment 1)

Course assignment: train the SAME logistic regression three ways (scikit-learn, manual PyTorch
loop, `torch.nn.Module` + `torch.optim`), compare against baselines, serve from a Gradio
dashboard that loads saved models and never retrains. Grading is a written comprehension
check without the code, so the owner must understand every line. See PLAN.md for the full plan.

## Business framing
Display Internacional has ~1,300 zero-stock SKUs in a ~2,100-SKU catalogue. Nobody can review
them all. The model scores each SKU with a probability of being **dead stock** and ranks them,
so the family reviews ~200 top candidates for discontinuation instead of the whole catalogue.

Cost assumptions (the owner's, stated as assumptions, values live in `src/data.py`):
- false positive = discontinuing a live item = `FP_COST` DOP
- false negative = keeping a dead item listed = `FN_COST_PER_YEAR` × `HORIZON_YEARS` = `FN_COST` DOP
  (FP is a one-off loss, FN recurs each year the listing survives, so both use one horizon)
The decision threshold is chosen to minimise expected cost `FP_COST·FP + FN_COST·FN` on the
validation slice, and the dashboard recomputes that cost live as the threshold moves.

## Hard constraints — do not relax these
- Target is binary: `dead = (Stock <= 0) & (Punto reorden == 0)`, on `Tipo == 'Producto'` rows
  with `Costo > 0`. `make_target` in `src/data.py` returns it as int. Nothing else defines it.
- **Leakage: NEVER use as features** `Stock`, `Estado stock`, `Valor inventario`,
  `Punto reorden`, `Cant. reorden`, `Alertas` — the label is built from these.
- **Price columns stay excluded**: `Precio detalle`, `Precio mayor 1/2/3`, `Precio mínimo`,
  `Margen %`. They are markup × cost and add nothing beyond `Costo`.
- `Costo` is allowed as a feature but currently NOT used: `log(Costo)` was tried and dropped
  (no AUC gain). It is still read, with `Precio detalle`, to price a false positive.
- Only logistic regression. No trees, no boosting, no neural nets beyond a single `nn.Linear`.
- Three models: `LogisticRegression(C)`; manual PyTorch loop with sigmoid + binary
  cross-entropy + L2 on raw tensors; `nn.Linear` + `BCEWithLogitsLoss` + `Adam`.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, plus expected business cost at the chosen
  threshold. Report test metrics at 0.5 and at the chosen threshold.
- Baselines: majority class, and "dead if Costo > train median".
- Tuning is limited to C / L2 strength and learning rate for the two PyTorch versions, chosen on
  the validation slice by ROC-AUC. No grid search over anything else. No AutoML.
- `train.py` is the only script that fits anything. `app.py` loads from `models/` and must fail
  loudly if a model file is missing rather than retraining.
- Same train/test split for all three models: `random_state=42`, 80/20, defined once in
  `src/data.py`. Scaler fit on train only.
- Keep it small: pandas, numpy, scikit-learn, torch, gradio, matplotlib. No extra deps
  without asking.

## Running
Dependencies come from `pyproject.toml` via uv (numpy 2.x, torch 2.4+); there is no requirements.txt.
```bash
uv run python main.py train    # runs train.py — the only stage that fits anything
uv run python main.py app      # runs app.py — Gradio dashboard, loads models/, never retrains
```
`main.py` is the graded entrypoint and refuses to run if both `<stage>.py` and `<stage>.ipynb` exist.

## Division of labour
The owner writes `src/features.py` (feature engineering) and the manual PyTorch loop in
`src/train_manual.py` himself — these are the comprehension-check targets. For those two files:
review, explain, point out bugs, quiz — do not write them wholesale unless explicitly asked.
Everything else (sklearn model, nn.Module model, saving/loading, Gradio app, plots) you may
generate, but keep it readable and commented so it can be explained from memory.

## Data
`data/inventario_julio_2025_limpio.csv` — 2,098 SKUs, July 2025 stock snapshot, cleaned.
Modelling subset: `Tipo == 'Producto'` and `Costo > 0` → 2,067 rows, of which 1,253 (60.6%)
are dead by the definition above. Currency DOP (assumed).
`Suplidor (ID)`, `Área (ID)`, `Dpto (ID)` are numeric codes; names are not available.
Column meanings are in `data/README.md`.

## Language
Code and comments in English. Column names stay in Spanish as in the CSV. REPORT.md in English.

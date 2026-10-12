# Display Internacional — SKU cost estimator (Assignment 1)

Course assignment: train the SAME linear regression three ways (scikit-learn, manual PyTorch
loop, `torch.nn.Module` + `torch.optim`), compare against a baseline, serve from a Gradio
dashboard that loads saved models and never retrains. Grading is a written comprehension
check without the code, so the owner must understand every line. See PLAN.md for the full plan.

## Hard constraints — do not relax these
- Target is `log(Costo)`. Never predict raw `Costo`.
- **Leakage: NEVER use as features** any price column (`Precio detalle`, `Precio mayor 1/2/3`,
  `Precio mínimo`, `Margen %`) — they are markup × cost. Nor any stock column
  (`Stock`, `Estado stock`, `Valor inventario`, `Punto reorden`, `Cant. reorden`, `Alertas`).
- Only linear regression. No trees, no boosting, no neural nets beyond a single `nn.Linear`.
- Tuning is limited to regularization strength (alpha / weight_decay) and learning rate for the
  two PyTorch versions. No grid search over anything else. No AutoML.
- `train.py` is the only script that fits anything. `app.py` loads from `models/` and must fail
  loudly if a model file is missing rather than retraining.
- Same train/test split for all three models: `random_state=42`, 80/20, defined once in
  `src/data.py`. Scaler fit on train only.
- Keep it small: pandas, numpy, scikit-learn, torch, gradio, matplotlib/plotly. No extra deps
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
Modelling subset: `Tipo == 'Producto'` and `Costo > 0` → 2,067 rows. Currency DOP (assumed).
`Suplidor (ID)`, `Área (ID)`, `Dpto (ID)` are numeric codes; names are not available.
Column meanings are in `data/README.md`.

## Language
Code and comments in English. Column names stay in Spanish as in the CSV. REPORT.md in English.

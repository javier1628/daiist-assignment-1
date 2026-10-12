# REPORT

## 1. Dataset
<!-- what it is, where it came from, size, cleaning done (see data/README.md) -->

`data/inventario_julio_2025_limpio.csv` — a July 2025 stock snapshot of Display Internacional,
2,098 SKUs (one row per SKU), already cleaned: duplicates removed, hidden negative stock recovered,
an empty location column dropped, descriptions normalised (see `data/README.md`). The modelling
subset keeps only physical products with a usable cost (`Tipo == 'Producto'` and `Costo > 0`) →
**2,067 rows**, split 1,653 train / 414 test. Costs are in DOP (assumed). `Suplidor (ID)`,
`Área (ID)` and `Dpto (ID)` are opaque numeric codes — no names are available — which shapes how we
encode them (§4).

## 2. Business problem
<!-- the decision this supports; who uses it; what "good enough" means -->

Display Internacional is a distributor of retail fixtures (hooks, hangers, shelving, checkouts,
showcases, mannequins): it **buys** items from suppliers at a landed **cost** and **sells** them to
retailers at a **markup**. Markup is the company's own policy, so cost is the one unknown that gates
every quote — `selling price ≈ cost × markup`. Estimating cost is therefore the valuable half: once
you have it, the price follows.

The model estimates landed cost from a SKU's attributes (description, code prefix, supplier, pack
size), which supports two daily decisions for the sales staff:

- **Quote before the supplier answers.** For a new SKU, an item not bought in years, or one whose
  supplier quote is still pending, nobody knows the cost yet. The estimate lets staff apply the
  standard markup and quote on the spot instead of making the customer wait days.
- **Sanity-check a supplier quote.** An independent "this should cost roughly X" flags quotes that
  look off (wrong item, wrong currency, a typo, a bad deal) *before* the company commits to buying.

Asking the supplier every time is exactly what fails in these cases — no history, no response yet —
so the model fills the gap from what is already known about the product.

**"Good enough":** this is a **guardrail, not a price oracle.** Being within ~±40% is useful, and the
bands the dashboard shows matter more than the point estimate. Nobody sets the final price straight
from the model — it is a fast ballpark and an error-detector, not the quote itself.

**How it is used.** Delivery is the Gradio dashboard (`app.py`), which loads the saved models from
`models/` at startup and never retrains. On the Estimator tab a salesperson types what they know
about a SKU — description, code prefix, department, supplier, pack size — and gets three cost
estimates (Ridge, manual, nn.Linear) side by side, each with a low–high band, in DOP. They then
apply the standard markup to produce a quote, or compare the estimate against a supplier's quote to
see if it is in the right ballpark. The Data and Models tabs are for an analyst/manager to inspect
the catalogue and judge how far to trust the tool before relying on it; they are not part of the
daily flow.

## 3. Target and leakage
<!-- why log(Costo); why every price column and stock column is excluded -->

**Target = `log(Costo)`, never raw cost.** Cost spans ~0.08 → 3.2M DOP. On the raw scale a linear
model chases the handful of giant items — an error of a few thousand pesos on a 500k showcase
dwarfs the entire cost of a 50-peso hook, so the fit ignores the cheap items that make up most of
the catalogue. Modelling the log fixes two things: (a) it makes *relative* error the thing we
minimise, so "off by 2×" costs the same whether the item is cheap or expensive; (b) size and markup
effects are multiplicative in pesos but become *additive* in log space, which is exactly the shape a
linear model can fit. We report error both in log space (what we optimise) and back-transformed to a
DOP % error (what the business feels).

**Leakage — price and stock columns are banned as features.** Every price column
(`Precio detalle`, `Precio mayor 1/2/3`, `Precio mínimo`, `Margen %`) is literally markup × cost, so
it *contains the target*. A model given price would look near-perfect in testing and be useless in
the real job — estimating the cost of a SKU whose price isn't set yet (new item, pending quote).
Stock columns (`Stock`, `Valor inventario`, reorder points, alerts) are a July-2025 snapshot of
inventory levels that has nothing to do with what the item costs to buy; including them would just
let the model memorise noise. Both groups are listed in `src.data.LEAKY_COLUMNS` and never read.

## 4. Features
<!-- what you built from the description and code; what you tried that didn't help -->

### Frequency encoding instead of full one-hot for IDs
`Suplidor (ID)`, `Área (ID)` and `Dpto (ID)` are opaque numeric codes with many levels (≈74
suppliers). Full one-hot on ~1,650 training rows would add dozens of mostly-empty columns, each with
only a handful of examples to estimate a weight from — the model would fit per-supplier noise rather
than a real effect. Instead we **frequency-encode**: replace each ID with how often it appears in the
*train* set, i.e. one number per column that still carries signal ("is this a high-volume supplier?")
without the column explosion. Departments additionally get a top-8 one-hot because a few dominant
departments are worth modelling explicitly; everything beyond the top-N collapses to an `OTHER`
bucket so train and test always produce the same columns.

### Why Ridge (penalising large coefficients)
All three models add an L2 penalty on the coefficients (`alpha · Σw²`) on top of the squared
error — i.e. Ridge, not plain least squares. The penalty acts on the **weights**, not the errors:
it taxes large coefficients so no single feature can dominate the estimate. Business reasons:

- **Stable guardrail.** The tool sanity-checks quotes to ~±40%. A huge weight makes the estimate
  hostage to one input (a parsed size, a supplier-ID frequency) — a small wobble swings the number
  wildly. Shrinking weights keeps the estimate smooth and hard to knock off a cliff.
- **Generalises to unseen SKUs.** With ~2,000 rows and 46 noisy features (parsed text, opaque IDs),
  unpenalised regression over-commits to training-set coincidences (e.g. a rare code prefix on three
  expensive items) that won't hold on the new SKU typed in tomorrow — which is the whole use case.
- **Tames correlated ID features.** Supplier / dept / area codes overlap; without a penalty,
  regression answers correlation with wild offsetting weights. Ridge spreads influence sensibly.

Cost: regularisation adds a little bias (the model is nudged flatter), trading a touch of training
accuracy for stability on unseen data — worth it for a guardrail. Here Ridge only modestly beat the
baseline, so it is doing more to keep the model sane than to win big on accuracy.

**Ridge over Lasso / Elastic Net.** Three reasons, in order of weight for this project:
(1) Ridge's L2 penalty `Σw²` is smoothly differentiable, so the manual gradient-descent loop works
and all three methods fit the *identical* model; Lasso's `Σ|w|` has a kink at zero that plain
gradient descent can't handle (needs proximal/coordinate methods). (2) Our ID features are
correlated — Ridge keeps and spreads them, whereas Lasso arbitrarily zeros one of a correlated
group and is unstable across resamples. (3) We have only 46 deliberate features, so Lasso's
feature-selection (its main advantage) buys us nothing. Elastic Net just adds Lasso's kink plus a
second hyperparameter to tune — against our keep-it-simple scope — for no payoff here.

## 5. Split and evaluation
<!-- random 80/20 and why a time split was impossible; metrics chosen and why -->

**Random 80/20 split (`random_state=42`), defined once in `src.data`.** A time-based split (train on
older SKUs, test on newer) would be the gold standard for a forecasting-style tool, but the data is a
single July-2025 snapshot with no dates, so it is simply not possible — stated as a limitation. The
one split is created once and reused by all three models so the comparison is fair.

**Nothing is learned from the test set — not even the scaler.** Everything the pipeline learns from
data (the `StandardScaler` stats, the frequency tables, the NaN-fill medians, the top-N category
lists) is computed on *train only* and stored in the `fit` dict; test rows and dashboard input reuse
that dict and learn nothing. This is why `build_features` has the `fit=None` (learn) vs `fit=…`
(reuse) split. If the scaler or frequency tables had seen the test rows, information would leak
backward and the test score would be optimistic.

**Hyperparameters are tuned on a validation slice of train, not on test.** To pick Ridge's `alpha`
and the manual loop's `lr`, we carve a validation set out of the *training* data, choose the setting
that scores best there, refit on the full training set, and touch the test set exactly **once** at
the end. If we had chosen `alpha` by what scored best on test, the test number would no longer be an
honest estimate of performance on unseen SKUs — we'd have quietly fitted to it.

**Metrics: log-space and DOP.** We report RMSE and MAE in log space (the quantity the models
actually minimise) and a **median** absolute % error back in DOP (what a salesperson experiences).
We use the *median*, not the mean, % error because a few stale near-zero costs (items with cost < 1
DOP) would blow up a mean percentage error and misrepresent typical performance.

## 6. Three-method comparison

All numbers are on the held-out test set (414 SKUs). RMSE/MAE and R² are in log space; median % error
is back in DOP.

| Method | RMSE (log) | MAE (log) | Median abs % error (DOP) | R² | Notes |
|---|---|---|---|---|---|
| Baseline: mean | 2.183 | 1.788 | 90.3% | −0.02 | predict train mean log-cost |
| Baseline: mean per prefix | 1.994 | 1.502 | 84.1% | 0.15 | code prefix alone carries real signal |
| scikit-learn Ridge (α=10) | 1.726 | 1.293 | 77.6% | 0.36 | |
| PyTorch manual loop (lr=0.01) | 1.721 | 1.288 | 77.7% | 0.36 | L2 = α/n |
| PyTorch nn.Module + Adam | 1.729 | 1.295 | 77.6% | 0.36 | weight_decay = α/n |

**Do the three agree? Yes — essentially identical.** All three share the same intercept to four
decimals (6.6214) and land within a small coefficient distance of each other (‖ridge−torch‖≈0.13,
‖ridge−manual‖≈0.31, against a coefficient norm of ≈1.44), and their test metrics differ only in the
third decimal. This is the expected result: they are the *same* linear model fit three ways, so once
the L2 penalty is matched (`l2 = α/n`, §4/§3) the tiny remaining gaps are just optimiser and
convergence noise, not real modelling differences. The point of training it three ways is to
*understand* the method, not to gain accuracy — the methods are interchangeable here.

## 7. What didn't work

Findings from the actual run (not speculation):

- **Tuning the regularisation strength barely moved anything.** On the validation split, Ridge at
  α = 0.1, 1 and 10 all scored RMSE ≈ 1.647 — a flat curve. α = 10 was marginally best (and the
  most stable) so we kept it, but regularisation strength is clearly not the bottleneck.
- **The optimisation method made no practical difference.** sklearn, the manual loop and
  nn.Linear+Adam gave the same coefficients and the same metrics (§6). Swapping methods is a
  comprehension exercise, not a lever on accuracy.
- **The features only modestly beat the baselines.** Ridge reaches R² ≈ 0.36, i.e. ~64% of
  log-cost variance is still unexplained, and the code-prefix baseline alone already gets R² ≈ 0.15.
  The crude keyword parse (family/material/size) leaves most of the description unused — many rows
  fall into `OTHER` and many sizes are NaN and filled with the median — so the signal is thin.

### Future improvements (identified, not yet implemented)
Ranked by expected impact; all stay within the linear-only constraint (feature work is unrestricted):

1. **Clean the label noise** — drop/flag stale costs (< 1 DOP); bad targets inflate RMSE directly.
2. **Bag-of-words / TF-IDF on the description** — a linear model on text is still linear regression,
   and the description is the richest unused signal. Likely the biggest single gain. (This adds many
   columns, which makes Ridge's L2 penalty more important, not less.)
3. **Target-mean encode supplier/dept** — captures whether a supplier sells *expensive* items, which
   frequency encoding (how *common* it is) cannot. Must be computed within train folds only, or it
   leaks.
4. **Parsed volume + a few interaction terms** (e.g. size × material) — a linear model cannot form
   products of features on its own.
5. **Per-segment prediction bands** — report family-specific RMSE so confident segments get tighter
   bands instead of one global ÷5–×5 band.

The genuinely largest accuracy lever — a non-linear model (gradient boosting) — is out of scope for
this linear-only assignment (see §8).

## 8. Limitations

**The prediction band is wide.** Error is measured in log space (test RMSE ≈ 1.72), and converting
back to pesos makes a ±1 RMSE band *multiplicative*: roughly ÷5 to ×5 around the point estimate. So
in practice the tool indicates an order-of-magnitude range ("hundreds to low thousands"), not a
precise figure — useful as a sanity-check that catches a quote that is 10× off, but not precise
enough to set a quote from. The band is wide because the models only modestly beat the baselines
(1.72 vs 1.99 RMSE), which says the signal is thin in the current features, not that training is
broken. Narrowing it needs richer features (e.g. bag-of-words/TF-IDF on the description, parsed
volume, target-encoded supplier/dept) and cleaner cost labels — not more model tuning. Part of the
spread is also irreducible: supplier negotiation, import timing and raw-material prices drive cost
and are absent from the data entirely, so no feature work makes the band tight with this dataset.

## 9. How to run

```bash
uv sync                                # dependencies from pyproject.toml (numpy 2.x, torch 2.4+)

uv run python main.py train            # fits all models, writes artefacts to models/
uv run python main.py app              # launches the Gradio dashboard (loads models/, no retraining)
```

`train.py` is the only script that fits anything; it writes `scaler.joblib`, `feature_fit.joblib`,
`ridge.joblib`, `manual.pt`, `torch_module.pt`, `feature_names.json`, `metrics.json` and
`test_predictions.csv` into `models/`. `app.py` loads those at startup and fails loudly if any are
missing, rather than retraining. (Note: `app.py` is not implemented yet — `train.py` and its
artefacts are complete and runnable now.)

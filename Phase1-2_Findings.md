# Big Chalk Capstone — Phase 1 & 2 Findings
### Data understanding + first end-to-end automated model
*Team: Feifan Liu, Boqi Niu, Jiahao Li · Prepared for the Big Chalk check-in (week of June 29, 2026)*

---

## 1. Executive summary

We have completed Phase 1 (understand the data) and a first cut of Phase 2 (build one
automated model end to end). Working on **Brand 1 × Channel 1**, an automated,
sign-constrained marketing-mix regression explains **84% of weekly Volume Sales
variance** (adjusted R² 0.83, in-sample MAPE 11.7%, 26-week out-of-sample MAPE 19.6%)
using **10 automatically selected predictors**, every one of which respects its
expected business sign and all with VIF < 2.

The same pipeline runs unchanged on other slices with strong fit (Brand 2 × Ch1 R²=0.90,
Brand 3 × Ch1 R²=0.92, Brand 4 × Ch1 R²=0.85), which gives us confidence it will scale to
all Brand × Channel combinations in Phase 3.

The code is already modular and reusable (`code/capstone_pipeline.py`), so the Phase 3
expansion and Phase 4 dashboard can call the same functions rather than re-implementing.

---

## 2. The data (Phase 1)

- **Structure:** one workbook, a *General Data Dictionary* tab plus one tab per brand
  (**Brand 1–10**). Grain is **Channel (Geography) × Brand (Product) × Week**.
- **Coverage:** ~3 years of weekly data, **156 weeks, Jan 2023 → Dec 2025**; up to 9
  channels per brand; ~1,248 rows and **113 columns** per brand sheet.
- **Quality:** the modeled slice (Brand 1 × Ch1) has **no missing values**. Minor quirks
  exist elsewhere (Brand 6 has some `#N/A` channels; one brand sheet carries a numeric
  column header) — the pipeline now handles these defensively.

**Predictor families** (per the dictionary): own price, category/competitive price, own
distribution (ACV, items/store, TPD), trade execution (Weighted Weeks by merch
condition), 12 media-spend channels, macro-economic series (~18), seasonality & trend,
and Competitor 1–6 distribution/trade.

### Key Phase 1 finding — target leakage
The columns most correlated with Volume Sales are **not usable predictors**: `Dollar Sales`
(= Price × Volume Sales) and the `Volume Sales <merch condition>` columns are algebraic
**decompositions of the target itself**. Including them would inflate fit while teaching
the model nothing causal. We **exclude all `Volume Sales *` and `Dollar Sales *` columns**
from the predictor set and model only genuine marketing-mix drivers. This is the single
most important modeling decision in this phase.

---

## 3. Methodology (Phase 2)

The engine is a deliberate two-stage design driven by Alex's requirement that final-model
coefficients be constrainable (positive / negative / unconstrained / bounded away from 0):

1. **Feature engineering**
   - **Media adstock:** each media-spend series is geometrically decayed
     (`a_t = x_t + λ·a_{t-1}`, λ=0.5 default) to capture advertising carryover.
   - **Sign priors by family:** price → negative; own distribution, trade, media,
     seasonality → positive; competitive variables → negative; macro → unconstrained.

2. **Automated variable selection (build phase)**
   - VIF pruning (drop predictors with VIF > 10) to control collinearity, then
   - forward stepwise entry at p < 0.05 on standardized predictors.

3. **Constrained final fit**
   - Bounded least squares (`scipy.optimize.lsq_linear`) enforces the family sign priors
     directly. This is why we **do not use Ridge for the final model** — Ridge cannot honor
     sign/zero-exclusion bounds — exactly the limitation Alex flagged. Ridge/Lasso/tree
     importance remain available as *exploration* tools.
   - **Dead-predictor pruning:** any predictor driven to its zero bound is removed and the
     model refit, so the delivered model contains only contributing drivers.

4. **Outputs the engine produces for every model**
   - Coefficients with expected sign and family, OLS t-stats and VIF (diagnostics),
   - fit metrics (R², adjusted R², in-sample & holdout MAPE, Durbin-Watson),
   - **additive contributions ("due-tos")** that reconstruct the prediction exactly
     (verified reconstruction error ≈ 1e-10),
   - actual-vs-fitted, residual, and driver-importance charts.

5. **Validation:** a **26-week time-based holdout** (train on the first 130 weeks, score
   the last 26) — appropriate for time series, unlike random k-fold.

---

## 4. Results — Brand 1 × Channel 1

| Metric | Value |
|---|---|
| Predictors selected | 10 |
| R² | 0.838 |
| Adjusted R² | 0.827 |
| In-sample MAPE | 11.7% |
| Holdout MAPE (26 wks) | 19.6% |
| Durbin–Watson | 0.81 |
| Sign conflicts | 0 |
| Max VIF | < 2 |

**Drivers (all signs as expected):** trade execution (Weighted Weeks Display, Weighted
Weeks Feature) and Seasonality lift volume; competitor display and competitor price
reductions suppress it; Meta and Google ad spend contribute positively. Full coefficients
are in `outputs/model_coefficients.csv`; charts are in `outputs/`.

**Benchmark:** holdout MAPE of 19.6% vs ~92% for a naive train-mean baseline — the model
adds large, real predictive value.

---

## 5. Open issues & honest caveats (for discussion)

1. **A macro level series is absorbing the baseline.** `SNAP_Participants` enters with a
   tiny coefficient but, because its raw magnitude is huge (tens of millions), its
   contribution effectively sets the model's base level alongside a large negative
   intercept. Fit is fine, but the *due-to* story is distorted. Options: drop raw macro
   *level* series, index/normalize them, or first-difference. (Our charts show a
   *demeaned* contribution view to avoid this visual artifact.)
2. **Residual autocorrelation** (Durbin–Watson ≈ 0.81) — expected for weekly sales but it
   biases standard errors. Phase 3 should consider lagged dependent terms or
   Newey–West / HAC standard errors.
3. **In-sample vs holdout gap** (11.7% → 19.6%) suggests mild over-fit or a level shift in
   2025; worth checking whether 2025 volumes trend differently.

---

## 6. Questions for Big Chalk

1. How does Big Chalk formally define a **"due-to" / contribution** — additive in levels
   (as we've built), or share-of-prediction, or vs a base period?
2. Preferred **adstock decay rates** per media channel, or should we fit them?
3. How should **raw macro level series** (e.g. SNAP counts) be treated — included as-is,
   normalized, or excluded as forecasting overlays only?
4. Are there **must-include** drivers the client always expects to see (e.g. own price,
   distribution) even when automated selection drops them?
5. What **fit thresholds** does the client consider acceptable (R² / MAPE)?
6. For the Phase 4 UI: one **combined upload file** for all Brand × Channel, or
   per-brand files plus an optional spec file for constraints?

---

## 7. How this seeds Phases 3 & 4

- **Phase 3 (expand to all models):** `run_slice()` already takes any brand/channel and
  returns a complete, validated model object. A loop over all Brand × Channel combinations
  plus a results table is a small step from here.
- **Phase 4 (UI):** the dashboard can call `assemble_matrix` → `forward_stepwise` →
  `constrained_fit` and expose the sign-prior table and adstock decay as user controls,
  surfacing the same coefficients, contributions, and fit charts we already generate.

---

## 8. Files

| File | What it is |
|---|---|
| `code/capstone_pipeline.py` | Reusable engine (load, adstock, selection, constrained fit, diagnostics, contributions) |
| `code/run_brand1_channel1.py` | Phase 1+2 runner for Brand 1 × Channel 1 |
| `outputs/eda_01..04_*.png` | EDA: target over time, seasonality, distribution, predictor correlations |
| `outputs/model_01..03_*.png` | Actual-vs-fitted, residual diagnostics, driver importance |
| `outputs/model_coefficients.csv` | Final coefficients, signs, t-stats, VIF, contributions |
| `outputs/model_summary.csv` | Headline fit metrics |
| `Project_Brief.md` | Overall project brief (scope, data dictionary, timeline) |

# Big Chalk Capstone — Findings & Methodology (living document)
*Team: Feifan Liu, Boqi Niu, Jiahao Li · Updated July 6, 2026 (post sponsor review)*
*Week-2 snapshot preserved separately in `Phase1-2_Findings.md`.*

---

## 1. Where the project stands

Phases 1–3 of the project scope are complete; Phase 4 (UI) is functional with the
core scope requirements implemented. Across **all 82 Brand × Channel combinations**:
80 models built (2 slices correctly skipped — brand not sold in that channel),
**median R² 0.89, median holdout MAPE ~10%**, full batch runtime **~9 seconds**.
The Dash UI loads any client datafile, auto-detects product sheets, exposes
per-variable modeling controls, and can batch-run every combination from the browser.

---

## 2. Phase 1 — data understanding (unchanged conclusions)

Grain: Channel × Brand × Week; 156 weeks (Jan 2023 – Dec 2025); 10 brand sheets;
~113–123 columns each. The critical finding stands: `Volume Sales *` and
`Dollar Sales *` columns are algebraic decompositions of the target and are excluded
as predictors (now enforced dynamically for whatever target is chosen).

New since week 2: Brand 2's media columns are named `Brand 0210_*` — an anonymization
quirk that validated the config-driven design (absorbed with zero code change).

---

## 3. Phase 2 — the engine, after the July 6 sponsor review

Alex's review reshaped five things; all are implemented and re-validated:

1. **Adstock normalization.** Changed to the Big Chalk convention
   `a_t = (1−λ)·x_t + λ·a_{t−1}` so total adstocked ≈ total raw impressions (the
   naive form inflates totals). Decay is per-media-variable (search ≈ 0.2,
   CTV/TV ≈ 0.7, default 0.5 = industry standard); a totals check runs every model.
2. **Config-table variable mapping.** No hard-coded variable names anywhere. Each
   variable gets family, expected sign, optional custom coefficient bounds, adstock
   decay, and a role: `auto` (selection decides), `force` (client-mandated — bypasses
   VIF and pruning), `exclude`.
3. **Configurable target** carried through the entire pipeline, charts, and exports.
4. **Set-year window:** latest 104 weeks (52/156 configurable) with a time-based tail
   holdout; MAPE computed over non-zero actuals only.
5. **Contribution reporting:** intercept as its own line; signed due-to sums by model
   year with YoY change; average weekly contributions computed only over weeks with
   execution for media.

**Method equivalence:** our `lsq_linear` (TRF) estimator reproduces the sponsor's
production `curve_fit` (TRF) approach to a max relative coefficient difference of
**3.1e-07** on the Brand 1 × Channel 1 model (`code/reference_alex_curvefit.py`).

**Method choices (for the record):** bounded least squares over Ridge (Ridge cannot
honor sign/box constraints); forward stepwise over Lasso (transparent, each entry a
testable p-value decision); VIF pruning over pairwise correlation/PCA (catches
multi-variable redundancy; keeps explainable raw variables); time-based holdout over
random k-fold (weekly series are autocorrelated — random folds leak the future).

---

## 4. Phase 3 — all Brand × Channel combinations

`run_all.py` batches every slice: **80 models, median R² 0.89, median holdout MAPE
~10%**, 8 slices above 40% holdout MAPE (structural-change candidates), 2 skipped
(all-zero target). Results in `outputs/all_models_summary.csv` + per-slice tables.

**Performance:** selection was rewritten to exact fast equivalents — VIFs from the
diagonal of the inverse correlation matrix (one inversion instead of k auxiliary
regressions) and stepwise p-values from a direct normal-equations solve. ~5× faster
per slice (2.7s → 0.55s), identical selections and coefficients (verified to 1e-6).
Full batch: ~90s → **~9s**.

### Case study: the Brand 1 × Channel 1 holdout gap
In-sample MAPE ~10% but holdout (last 13 weeks, Q4 2025) error was **73%** — every
prediction biased high. Diagnosis: volume fell from a 87K/week quarterly average
(Q4 2024) to 35K (Q4 2025) while **distribution collapsed** (TPD −24%, ACV −15%),
and the automated selection had dropped every distribution variable (VIF prune
casualty; once excluded, ACV looks insignificant in-sample, p = 0.93, despite
carrying the out-of-sample structural signal). **Forcing `ACV Weighted Distribution`
into the model cut holdout MAPE from 73% → 32%** and raised R² 0.798 → 0.828.
This is precisely the "client-mandated variable" scenario the sponsor described —
statistical selection criteria and business necessity can disagree, and the engine
now lets the analyst overrule the statistics. The remaining ~32% suggests a
late-2025 event not in the data (delisting?) — flagged for the sponsor.

---

## 5. Phase 4 — the UI (Dash)

Four tabs: **Run & diagnostics** (fit metrics, actual-vs-fitted with holdout
forecast, residuals, coefficients), **Variable controls** (editable per-product
config: signs, bounds, decays, roles — with validation), **Contributions** (due-tos
by model year, YoY, execution-masked weekly averages), **Batch overview** (all
slices, sortable, red-flagged high-MAPE rows, click-through to any slice).

Scope requirements met: loads **any datafile** from the UI (product sheets
auto-detected — `Brand *` names or any sheet with Geography/Time columns; targets
discovered from the data), **Run all combinations** batches the loaded file using
the saved configs, drill-down / modify / rerun individual models all supported.
Validated on a synthetic "new client" workbook with alien sheet and column naming.

**Redesign (July 9):** the UI now implements the "Bench" direction from a Claude
Design handoff — IBM Plex type, chalk-navy accent, left workspace rail, metric
strip with REVIEW flags, family color chips, batch health dots (`assets/bench.css`).
Configs became per-dataset + per-product so two clients' files can't collide.
Implementation surfaced two real-data bugs worth recording: Brand 2 carries a
column literally named `0` (numeric header crashed target discovery — now sorted
with `key=str`), and an empty Dash component is *falsy*, which silently dropped a
subtitle from the layout and made the browser disable the Run callback with no
error. The dashboard now runs a startup wiring audit that fails loudly instead.

Remaining UI gap (known): per-slice (single Brand × Channel) config overrides —
configs currently apply per product across its channels.

---

## 6. Open issues & questions for the sponsor

1. **Brand 1 Q4 2025:** is there a known delisting/assortment event? (Would motivate
   an event-dummy/overlay mechanism, which the engine does not yet have.)
2. **Trade coefficients:** should volume-on-promo variables be hard-bounded < 1
   (subsidized-base logic)? The bounds machinery exists; policy decision pending.
3. **Adstock decays:** keep priors (search 0.2 / CTV 0.7) or fit via Optuna trials?
4. **Deployment:** where will the dashboard run at Big Chalk (analyst laptops,
   internal server)? Affects packaging.
5. **Residual autocorrelation** (DW < 1 on several slices): acceptable for due-to
   work, or should Phase 4+ add HAC standard errors / lagged terms?

---

## 7. File map

See README.md — layout, quickstart, and performance notes are maintained there.

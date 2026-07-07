# Big Chalk × Northwestern Capstone — Automated Regression Engine

Automated marketing-mix regression engine that models weekly **Volume Sales** (or any
chosen target) for every **Brand × Retailer-Channel** combination, with a Plotly Dash
UI for running and steering models. Sponsored by **Big Chalk Analytics**
(NU MSDS Summer 2026 Capstone).

**Team:** Feifan Liu · Boqi Niu · Jiahao Li (Edison)

## ⚠️ Confidential data — not in this repo
The client dataset (`Anonymized Data for Project.xlsx`), the kickoff deck, and meeting
transcripts are covered by the Big Chalk NDA and are intentionally **excluded** via
`.gitignore`. Obtain them through the team's secure channel and place the workbook in
the repo root before running.

## Repository layout
```
code/
  capstone_pipeline.py        Core engine: load, adstock, selection, constrained fit, contributions
  run_brand1_channel1.py      Single-slice runner (EDA + model + charts + exports)
  run_all.py                  Phase 3 batch runner: every Brand × Channel (~9 s for all 82)
  dashboard.py                Phase 4 Dash UI: diagnostics · variable controls · contributions · batch overview
  reference_alex_curvefit.py  Sponsor's curve_fit approach + equivalence test (matches to 3e-7)
configs/                      Per-brand variable configs edited via the dashboard
variable_config.csv           Brand 1 config used by the single-slice runner
outputs/                      Generated charts and result CSVs (derived, anonymized)
  all/                        Per-slice coefficient & contribution tables from run_all
  all_models_summary.csv      Cross-slice summary (R², MAPE, top drivers), sorted by holdout MAPE
requirements.txt              pip install -r requirements.txt
Project_Brief.md              Scope, data dictionary, timeline
Phase1-2_Findings.md          Methodology, results, open questions for the sponsor
```

## Quickstart
```bash
pip install -r requirements.txt
# place "Anonymized Data for Project.xlsx" in the repo root, then:
cd code
python run_brand1_channel1.py     # one slice, full EDA + diagnostics
python run_all.py                 # all brands × channels -> outputs/all_models_summary.csv
python run_all.py 1 3             # or in resumable chunks (Brand 1..3)
python dashboard.py               # UI at http://127.0.0.1:8050
```

## Approach (summary)
1. **No leakage** — decompositions of the target (`Volume Sales *`, `Dollar Sales *`) are excluded as predictors, dynamically for whatever target is chosen.
2. **Config-driven framing** — a per-brand variable config maps each column to a family, expected sign, optional custom coefficient bounds (e.g. 0.10–0.25), a per-variable adstock decay, and a role (`auto` / `force` = client-mandated, bypasses selection / `exclude`). No variable names are hard-coded — the engine absorbed Brand 2's `Brand 0210_*` naming quirk with zero code change.
3. **Two-stage estimation** — automated selection (VIF prune + forward stepwise) for the build, then **sign/bound-constrained least squares** for the final model (Ridge can't honor sign bounds; stepwise is more transparent than Lasso; VIF catches multi-variable redundancy that pairwise correlation misses). Verified numerically equivalent to the sponsor's production `curve_fit` approach (`reference_alex_curvefit.py`).
4. **Normalized adstock** — `a_t = (1−decay)·x_t + decay·a_{t−1}` so adstocked totals ≈ raw totals (never inflated); decay customizable per media variable (search ≈ 0.2, CTV/TV ≈ 0.7, default 0.5). A totals check prints each run.
5. **Set-year window** — model on the latest 104 weeks (52/156 configurable) with a **time-based tail holdout**; MAPE computed over non-zero actuals only.
6. **Contributions** — additive due-tos with the intercept as its own line; signed **sums by model year with YoY change**; weekly averages computed **only over weeks with execution** for media.

## Performance
Selection is exact but fast: VIFs come from the diagonal of the inverse correlation
matrix (one inversion instead of k auxiliary regressions) and stepwise p-values from a
direct normal-equations solve — ~10× faster than the statsmodels-per-candidate loop,
identical selections and coefficients (verified to 1e-6). Full 82-slice batch: ~9 s.
Slices are embarrassingly parallel if further scale is ever needed.

## Phase 3 results (all brands × channels)
80 models (2 slices skipped — brand not sold in channel): median R² 0.89, median
holdout MAPE ~10%. See `outputs/all_models_summary.csv`; high-MAPE slices flag
structural change (e.g. Brand 1 × Channel 1's distribution collapse — forcing
`ACV Weighted Distribution` into the model cut its holdout MAPE from 73% to 32%).

## Changes from sponsor review (Alex, 2026-07-06)
Adstock normalization + per-media decays · config-table variable mapping · custom
coefficient bounds · force-include/exclude lists · configurable target · 104-week
set-year window · due-tos by model year · execution-masked contribution averages.

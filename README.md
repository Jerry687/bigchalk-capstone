# Big Chalk × Northwestern Capstone — Automated Regression Engine

Automated marketing-mix regression engine that models weekly **Volume Sales** for every
**Brand × Retailer-Channel** combination, with a planned UI for running and tweaking models.
Sponsored by **Big Chalk Analytics** (NU MSDS Summer 2026 Capstone).

**Team:** Feifan Liu · Boqi Niu · Jiahao Li (Edison)

## ⚠️ Confidential data — not in this repo
The client dataset (`Anonymized Data for Project.xlsx`) and the kickoff deck are covered by
the Big Chalk NDA and are intentionally **excluded** via `.gitignore`. Obtain them through
the team's secure channel and place the workbook in the repo root before running.

## Repository layout
```
code/
  capstone_pipeline.py        Reusable engine: load, adstock, selection, constrained fit, diagnostics, contributions
  run_brand1_channel1.py      Phase 1+2 runner for Brand 1 × Channel 1 (EDA + model + charts + exports)
variable_config.csv           EDITABLE mapping: variable → family, sign, coef bounds, adstock decay, role
outputs/                      Generated charts and result CSVs (derived, anonymized)
Project_Brief.md              Scope, data dictionary, timeline
Phase1-2_Findings.md          Methodology, results, open questions for the sponsor
```

## Quickstart
```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn openpyxl
# place "Anonymized Data for Project.xlsx" in the repo root, then:
cd code
python run_brand1_channel1.py
```

## Approach (summary)
1. **No leakage** — decompositions of the target (`Volume Sales *`, `Dollar Sales *`) are excluded as predictors, dynamically for whatever target is chosen.
2. **Config-driven framing** — `variable_config.csv` maps each column to a family, expected sign, optional custom coefficient bounds (e.g. 0.10–0.25), a per-variable adstock decay, and a role (`auto` / `force` = client-mandated, bypasses selection / `exclude`). No variable names are hard-coded; the same engine handles any data set.
3. **Two-stage estimation** — automated selection (VIF prune + forward stepwise) for the build, then **sign/bound-constrained least squares** for the final model (Ridge can't honor sign bounds; stepwise is more transparent than Lasso; VIF catches multi-variable redundancy that pairwise correlation misses).
4. **Normalized adstock** — `a_t = (1−decay)·x_t + decay·a_{t−1}` so adstocked totals ≈ raw totals (never inflated); decay customizable per media variable (search ≈ 0.2, CTV/TV ≈ 0.7, default 0.5). A totals check prints each run.
5. **Set-year window** — model on the latest 104 weeks (52/156 configurable) with a **time-based tail holdout**; the **target is configurable** (volume / dollar / unit sales) and flows through all charts and exports.
6. **Contributions** — additive due-tos with the intercept as its own line; signed **sums by model year with YoY change**; weekly averages computed **only over weeks with execution** for media.

### Changes from sponsor review (Alex, 2026-07-06)
Adstock normalization + per-media decays · config-table variable mapping · custom coefficient bounds · force-include/exclude lists · configurable target · 104-week set-year window · due-tos by model year · execution-masked contribution averages.

See `Phase1-2_Findings.md` for full results and the sponsor question list.

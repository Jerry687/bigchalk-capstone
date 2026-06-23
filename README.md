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
1. **No leakage** — `Volume Sales *` and `Dollar Sales *` are decompositions of the target and are excluded as predictors.
2. **Marketing-mix framing** — predictors grouped into families (price, distribution, trade, media, macro, competitive) with expected coefficient signs.
3. **Two-stage estimation** — automated selection (VIF prune + forward stepwise) for the build, then **sign/bound-constrained least squares** for the final model (Ridge can't honor sign bounds).
4. **Adstock** on media spend; **time-based holdout** for validation; additive **due-to** contributions.

See `Phase1-2_Findings.md` for full results and the sponsor question list.

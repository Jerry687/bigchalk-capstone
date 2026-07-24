# Big Chalk × Northwestern Capstone — Automated Regression Engine

Automated marketing-mix regression engine that models weekly **Volume Sales** (or any
chosen target) for every **Brand × Retailer-Channel** combination, with a Plotly Dash
UI for running, steering, and exporting models. Sponsored by **Big Chalk Analytics**
(NU MSDS Summer 2026 Capstone).

**Team:** Feifan Liu · Boqi Niu · Jiahao Li (Edison)

## ⚠️ Confidential material — not in this repo
The client dataset (`Anonymized Data for Project.xlsx`), all **decks/slides**
(`*.pptx`), and **meeting transcripts** are covered by the Big Chalk NDA and are
**excluded** via `.gitignore`. **Agent / AI-tooling files** used during development
(`AGENTS.md`, `CODEX_HANDOFF.md`, `.agents/`, `.codex/`, `.claude/`, `*.inspect.ndjson`)
are also gitignored — they are not part of the deliverable. NDA material lives in the
gitignored `local/` folder. Obtain the data through the team's secure channel and place
the workbook in the repo root before running.

## Repository layout
```
code/
  capstone_pipeline.py        Core engine: load · adstock · selection · constrained fit · contributions · config resolution
  run_brand1_channel1.py      Single-slice runner (EDA + model + charts + exports)
  run_all.py                  Batch runner: every Brand × Channel (~10 s for all 82)
  dashboard.py                Dash UI (Bench design): Diagnostics · Variables · Contributions · Model Runs · Export · Definitions
  assets/bench.css            Bench design system (IBM Plex, chalk-navy) from the Claude Design handoff
  reference_alex_curvefit.py  Sponsor's curve_fit approach + equivalence test (matches to ~2e-8)
configs/                      Two-tier variable configs (auto-created; editable via the dashboard)
  varconfig_<dataset>_<brand>.csv            Product DEFAULT (applies to all channels)
  varconfig_<dataset>_<brand>__ch_<channel>.csv   optional Product × Channel OVERRIDE (wins)
  variable_config_legacy.csv                 legacy single-slice seed (unused; kept for reference)
outputs/                      Generated charts and result CSVs (derived, anonymized)
  all/                        Per-slice coefficient & contribution tables from run_all
  all_models_summary.csv      Cross-slice summary (R², MAPE, grade), sorted by holdout MAPE
docs/                         Project docs (briefs, findings, meeting prep, plans, reports)
  Project_Brief.md · Findings.md · Phase1-2_Findings.md · Monday_Meeting_Prep.md
  Meeting_Runbook_Jul21.md · Action_Items_Alex_Meeting_2026-07-21.md
  Alex_Dashboard_Edits_Plan.md   feature plan/tracker for the July PowerPoint feedback (Groups 1–9)
local/                        NDA-only material kept out of git (decks, transcripts) — gitignored
requirements.txt              pip install -r requirements.txt
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
python reference_alex_curvefit.py # sponsor-equivalence check (run from anywhere)
```

## Approach (summary)
1. **No leakage** — decompositions of the target (`Volume Sales *`, `Dollar Sales *`) are excluded as predictors, dynamically for whatever target is chosen.
2. **Config-driven framing (two-tier)** — a variable config maps each column to a family, expected sign, optional custom coefficient bounds (e.g. 0.10–0.25), a per-variable adstock decay, and a role (`auto` / `force` = client-mandated, bypasses selection & VIF pruning / `exclude`). A **Product default** applies to every channel; an optional **Product × Channel override wins** for that channel. The same resolver (`cp.resolve_config_path`) is shared by the dashboard, `run_all.py`, and the single-slice runner, so all three agree. No variable names are hard-coded — the engine absorbed Brand 2's `Brand 0210_*` naming quirk with zero code change.
3. **Two-stage estimation** — automated selection (VIF prune + forward stepwise) picks the structure; the final model is a **sign/bound-constrained least squares** fit (Ridge can't honor sign bounds; stepwise is more transparent than Lasso; VIF catches multi-variable redundancy pairwise correlation misses). Verified numerically equivalent to the sponsor's production `curve_fit` approach on the reported model (`reference_alex_curvefit.py`, ~2e-8 of the largest coefficient's scale).
4. **Normalized adstock** — `a_t = (1−decay)·x_t + decay·a_{t−1}` so adstocked totals ≈ raw totals (never inflated); decay customizable per variable that has one (search ≈ 0.2, CTV/TV ≈ 0.7, default 0.5). A totals check prints each run.
5. **Set-year window + always-reserved validation** — model on the latest 52/104/156 weeks. A **validation model** selects the structure on the window ending *before* an always-reserved tail (up to 13 weeks) and is scored on that tail → the out-of-sample **holdout MAPE** (so a holdout exists even at a full-history window). The **reported model** refits that *same structure* on the full window incl. the tail → the coefficients/due-tos shown. The validation holdout is preserved as the reported metric — the all-data in-sample fit is never relabeled "holdout".
6. **Contributions** — additive due-tos with the intercept as its own line; signed **sums by model year with YoY change**; weekly averages computed **only over weeks with execution** for media. A Contributions time filter re-aggregates the avg-weekly view by period (entire / current or previous year / Q4 / latest 13 or 4 weeks / uploaded time-map) and supports period-vs-period comparison.

## Performance
Selection is exact but fast: VIFs come from the diagonal of the inverse correlation
matrix (one inversion instead of k auxiliary regressions) and stepwise p-values from a
direct normal-equations solve — ~10× faster than the statsmodels-per-candidate loop.
Full 82-slice batch: ~10 s (each slice now fits a validation + a reported model).
Slices are embarrassingly parallel if further scale is ever needed.

## The UI (Bench design)
Plotly Dash app implementing the "Bench" direction from a Claude Design handoff. A
startup wiring guard audits every callback against the layout (Dash silently disables
callbacks referencing missing components when debug is off — learned the hard way).
Loads **any client datafile** (sheets and targets auto-detected). Tabs:

- **Diagnostics** — metric strip (R², adj R², in-sample & holdout MAPE, DW, predictors), actual-vs-fitted with the out-of-sample forecast on the reserved tail, residuals, and a coefficient table with family chips, t-stat bars, and a **VIF** column (>10 flagged, non-blocking).
- **Variables** — the two-tier config editor. A **CONFIG scope** selector (Product default / this channel) with status line; per-variable sign / adstock / bounds / role; **coef-now + due-to** context columns with a Y1/Y2/Total (or mapped period) toggle; hide-excluded filter; **Download / Upload template** (`.xlsx`/`.csv`, atomic, "unlisted = excluded"); Save / Regenerate / **Reset to default** / **Run** (a Variables run persists edits, then jumps to Diagnostics).
- **Contributions** — due-to totals by model year, avg-weekly due-to per driver with a **period filter + "vs" comparison + uploadable time-map**, and a YoY table. Charts carry k/M data labels.
- **Model Runs** (batch) — every Product × Channel with a **Good / Moderate / Bad Model** grade (colored cell), review flags, per-slice **Run model** and **Run all combinations**; click a row to load that slice.
- **Export** — one Excel (for the current slice or all combinations) with **fit statistics** (+ green/amber/red grade), **fit structure** (= the upload-template format, so export → edit → re-upload), and **weekly due-tos** (actual by date).
- **Definitions** — plain-language definitions of every metric, the grade thresholds, roles, adstock, the two-tier config, and the window/holdout method (kept in sync with the code).

Run controls live only on **Variables** (tuning) and **Model Runs** (batch); hitting Run
redirects to Diagnostics for the same slice. All config writes go through one process
lock and a transactional (temp-file + rollback) commit.

## Batch results (all brands × channels)
~80 models (2 slices skipped — brand not sold in channel): median R² ≈ 0.89. Grades and
holdout MAPE per slice in `outputs/all_models_summary.csv`; high-MAPE slices flag
structural change (e.g. Brand 1 × Channel 1's distribution collapse — forcing
`ACV Weighted Distribution` into the model cut its holdout MAPE from 73% to 32%).

## Change history
- **Sponsor review (Alex, 2026-07-06):** adstock normalization + per-media decays · config-table variable mapping · custom coefficient bounds · force-include/exclude lists · configurable target · set-year window · due-tos by model year · execution-masked averages.
- **PowerPoint feedback (July):** the run-control restructure, batch grade labels, chart data labels, VIF column, Variables context columns, two-tier config, template up/download, Export tab, Definitions tab, final-model-on-all-data, and the Contributions time filter. Item-by-item plan and status in `docs/Alex_Dashboard_Edits_Plan.md`.

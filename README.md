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
  capstone_pipeline.py        Core engine: load · adstock · Hill saturation · selection · constrained fit · contributions · config resolution
  multilevel.py               Pooled and hierarchical models (one product, all channels stacked)
  saturation_curves.py        SatCurve port (execution vs ROI) + the weekly raw/decayed/saturated diagnostic
  rollup.py                   Total Brand / Total Channel aggregation
  dashboard.py                Dash UI (Bench design), 9 screens — see "The UI" below
  assets/bench.css            Bench design system (IBM Plex, chalk-navy) from the Claude Design handoff
  run_brand1_channel1.py      Single-slice runner (EDA + model + charts + exports)
  run_all.py                  Batch runner: every Brand × Channel (~17 s for all 76)
  reference_alex_curvefit.py  Sponsor's curve_fit approach + equivalence test (matches to ~2e-8)
  test_multilevel.py          37 numerical checks, incl. the sponsor's hierarchy spreadsheet
  test_dashboard_screens.py   23 end-to-end checks driving every screen's callbacks
configs/                      Two-tier variable configs (auto-created; editable via the dashboard)
  varconfig_<dataset>_<brand>.csv            Product DEFAULT (applies to all channels)
  varconfig_<dataset>_<brand>__ch_<channel>.csv   optional Product × Channel OVERRIDE (wins)
  variable_config_legacy.csv                 legacy single-slice seed (unused; kept for reference)
outputs/                      Experiment results worth keeping (derived, anonymized)
  all_models_summary.csv      Cross-slice summary (R², MAPE, grade), sorted by holdout MAPE
  all_models_skipped.csv      Slices not modeled, with the reason for each
  multilevel_comparison.csv   Unpooled vs pooled vs hierarchical, per channel, same holdout
  multilevel_summary_by_brand.csv   The same rolled up to one row per product
  curve_optimizer_experiment.csv    Why the media-curve optimizer ships switched off
  (per-slice tables from run_all land here too; regenerated on every run, not tracked)
docs/                         Deliverables
  BigChalk_Mix_Engine_User_Guide.docx/.pdf   how to use the dashboard, every button
  build_user_guide.py         regenerates the guide (figures in docs/img/)
  Media_Curves_Math.md        the adstock + Hill maths behind the media transforms
local/                        NDA-only material kept out of git (decks, transcripts) — gitignored
requirements.txt              pip install -r requirements.txt
```

## Quickstart
```bash
pip install -r requirements.txt
# place "Anonymized Data for Project.xlsx" in the repo root, then:
cd code
python dashboard.py               # UI at http://127.0.0.1:8050  <- start here
python run_brand1_channel1.py     # one slice, full EDA + diagnostics
python run_all.py                 # all brands × channels -> outputs/all_models_summary.csv
python run_all.py 1 3             # or in resumable chunks (Brand 1..3)
python multilevel.py "../Anonymized Data for Project.xlsx" "Brand 4"   # three levels, one product
```

Verification:
```bash
cd code
python test_multilevel.py         # 37 checks: the sponsor's spreadsheet, SatCurve, conservation
python test_dashboard_screens.py  # 23 checks: every screen's callbacks, end to end
python reference_alex_curvefit.py # sponsor-equivalence check (run from anywhere)
```

## Approach (summary)
1. **No leakage** — decompositions of the target (`Volume Sales *`, `Dollar Sales *`) are excluded as predictors, dynamically for whatever target is chosen.
2. **Config-driven framing (two-tier)** — a variable config maps each column to a family, expected sign, optional custom coefficient bounds (e.g. 0.10–0.25), a per-variable adstock decay, and a role (`auto` / `force` = client-mandated, bypasses selection & VIF pruning / `exclude`). A **Product default** applies to every channel; an optional **Product × Channel override wins** for that channel. The same resolver (`cp.resolve_config_path`) is shared by the dashboard, `run_all.py`, and the single-slice runner, so all three agree. No variable names are hard-coded — the engine absorbed Brand 2's `Brand 0210_*` naming quirk with zero code change.
3. **Two-stage estimation** — automated selection (VIF prune + forward stepwise) picks the structure; the final model is a **sign/bound-constrained least squares** fit (Ridge can't honor sign bounds; stepwise is more transparent than Lasso; VIF catches multi-variable redundancy pairwise correlation misses). Verified numerically equivalent to the sponsor's production `curve_fit` approach on the reported model (`reference_alex_curvefit.py`, ~2e-8 of the largest coefficient's scale).
4. **Media transforms: raw → adstock → scale → Hill.** Normalized adstock `a_t = (1−decay)·x_t + decay·a_{t−1}` so adstocked totals ≈ raw totals (never inflated); decay customizable per variable (search ≈ 0.2, CTV/TV ≈ 0.7, default 0.5). **Hill saturation** `H(x) = xˢ/(xˢ+kˢ)` supplies diminishing returns — adstock first because carry-over is about *when* an impression lands, saturation last because diminishing returns apply to accumulated pressure. Same order as Robyn and Meridian. Saturation is **off by default** (see `outputs/curve_optimizer_experiment.csv` for why: the optimizer improves its own CV score far more reliably than the untouched holdout). A `scale` column divides a variable before fitting so a coefficient reads "per 1,000 impressions" — units only, fitted values identical to 1.2e-10.
5. **Fixed calendar window + always-reserved validation.** The window is an **absolute date range** anchored to the latest week in the whole workbook, *identical for every slice*. It used to be the last N *rows* each slice happened to have, which meant a slice delisted in mid-2023 still produced a model — from 2023 data — that leaked into the exports as a disjoint chunk of history. A slice without enough data inside the window (≥52 weeks, ≥26 selling) is **not modeled**, and appears in a "Not modeled" table with the reason rather than vanishing. Within the window, a **validation model** selects the structure on the weeks *before* an always-reserved 13-week tail and is scored on it → the out-of-sample **holdout MAPE**. The **reported model** refits that *same structure* on the full window → the coefficients shown. The all-data in-sample fit is never relabeled "holdout".
6. **Contributions vs due-tos** — two different numbers, kept distinct: a **contribution** is a level (volume a driver accounts for *in* a period), a **due-to** is a change (contribution in B minus contribution in A). "Why am I down year on year" is the second. Additive, with the intercept as its own line; weekly averages computed **only over weeks with execution** for media; unequal-length periods compared per week automatically. **Impact / SD** (coefficient × standard deviation) is the comparable number across drivers — raw coefficients live on different supports and cannot be ranked against one another.
7. **Three modeling levels** (`multilevel.py`) — **unpooled** (one model per Product × Channel), **pooled** (one model per product, every channel's weeks stacked, one coefficient per predictor), and **hierarchical** (each channel's coefficient = the pooled one × an index from that channel's own unconstrained fit, capped and shrunk). Legitimate because these are models *of* a time series, not time-series models: adstock and saturation are applied within each channel before stacking, so the rows are independent. Two things the pooled level requires — **national predictors** (identical in every channel) are split by each channel's share of product volume so the pieces sum back to the national figure, and **channel intercepts** (mean-centring within channel) let channels that differ 30× in size share slopes without the coefficients absorbing a level gap. A λ dial runs from pooled (0) to full index (1) and is fit on the holdout.

## Performance
Selection is exact but fast: VIFs come from the diagonal of the inverse correlation
matrix (one inversion instead of k auxiliary regressions) and stepwise p-values from a
direct normal-equations solve — ~10× faster than the statsmodels-per-candidate loop.
Full batch: **76 models in ~17 s**, each fitting a validation model *and* a reported
model. All three levels for one product take ~3 s.

Results are **built once and cached**, keyed by (datafile, target, window, selection
settings). Every result screen filters that cache in ~140 ms and never refits. Beyond
speed this removes a real hazard: two screens cannot show numbers from two different
fits, because only one set of results exists. Slices are embarrassingly parallel if
further scale is ever needed.

## The UI (Bench design)
Plotly Dash app implementing the "Bench" direction from a Claude Design handoff. A
startup wiring guard audits every callback against the layout (Dash silently disables
callbacks referencing missing components when debug is off — learned the hard way).
Loads **any client datafile** (sheets and targets auto-detected). Tabs:

- **Diagnostics** — metric strip (R², adj R², in-sample & holdout MAPE, DW, predictors), actual-vs-fitted with the out-of-sample forecast on the reserved tail, residuals, and a coefficient table with family chips, t-stat bars, and a **VIF** column (>10 flagged, non-blocking).
- **Variables** — the two-tier config editor. A **CONFIG scope** selector (Product default / this channel) with status line; per-variable sign / adstock / bounds / role; **coef-now + due-to** context columns with a Y1/Y2/Total (or mapped period) toggle; hide-excluded filter; **Download / Upload template** (`.xlsx`/`.csv`, atomic, "unlisted = excluded"); Save / Regenerate / **Reset to default** / **Run** (a Variables run persists edits, then jumps to Diagnostics).
- **Contributions** — due-to totals by model year, avg-weekly due-to per driver with a **period filter + "vs" comparison + uploadable time-map**, and a YoY table. Charts carry k/M data labels.
- **High Level** — every Product × Channel rolled up to **Total Brand**, **Total Channel** or a grand total: overall fit, total contributions, YoY due-tos, and the member grid underneath. Aggregates model *output*, not a new model, so the decomposition stays exact (4e-15 across all 76). Reports WMAPE alongside MAPE, because at the total level "error volume ÷ actual volume" is the number that means anything, and warns when some slices don't cover the whole window.
- **Saturation** — the weekly picture of what the engine actually fed the regression: **raw vs decayed vs decayed+saturated**, with each transform's correlation to the target and the target overlaid. Sliders for decay / midpoint / slope redraw everything live from the cached slice (no refit — the coefficient is held while the input's shape changes, and the screen says so). Below, the **execution-vs-ROI curve** ported from the sponsor's own `SatCurve` class, with the average/marginal crossover as the optimal spend level. A concave curve (slope < 1) has no crossover at all — provable from the Hill equation — and is reported as inconclusive rather than as advice to triple spend.
- **Multi-Level** — the same product modeled **unpooled / pooled / hierarchical**, all scored on the same per-channel 13-week holdout by WMAPE, with the winning level called out per channel. Shows which predictors were treated as national, the per-channel coefficient index, the λ-vs-holdout curve, and how well one shared set of coefficients serves each channel. Flags predictors whose index is unstable (mean coefficient smaller than its spread across channels).
- **Model Runs** (batch) — every Product × Channel with a **Good / Moderate / Bad Model** grade (colored cell), review flags, per-slice **Run model** and **Run all combinations**; click a row to load that slice. Slices deliberately not modeled appear underneath with the reason.
- **Export** — one Excel (for the current slice or all combinations) with **fit statistics** (+ green/amber/red grade), **fit structure** (= the upload-template format, so export → edit → re-upload), and **weekly due-tos** (actual by date).
- **Definitions** — plain-language definitions of every metric, the grade thresholds, roles, adstock, the two-tier config, and the window/holdout method (kept in sync with the code).

Run controls live only on **Variables** (tuning) and **Model Runs** (batch); hitting Run
redirects to Diagnostics for the same slice. All config writes go through one process
lock and a transactional (temp-file + rollback) commit.

## Batch results (all brands × channels)
**76 models**, median R² **0.881**, median holdout MAPE **11.5 %**. Six slices are not
modeled — too few weeks or too few selling weeks inside the fixed window — and are
listed with reasons in `outputs/all_models_skipped.csv` rather than silently dropped.
Per-slice grades in `outputs/all_models_summary.csv`.

High-MAPE slices flag structural change rather than a broken tool: Brand 1 × Channel 1
lost distribution part-way through the window, and forcing `ACV Weighted Distribution`
into the model is what makes it tractable (holdout MAPE 19.6 %).

Which modeling level wins is **not** constant, which is itself the finding
(`outputs/multilevel_comparison.csv`): across the 10 products, unpooled wins 7 and
hierarchical 3; across the 76 individual channels, unpooled 48, hierarchical 16, pooled
12. Pooling helps thin channels that were fitting noise on their own and costs the large
channel that had enough data to speak for itself. The fitted λ lands at 1 for four
products and at 0 for four others.

## Change history
- **Sponsor review (Alex, 2026-07-06):** adstock normalization + per-media decays · config-table variable mapping · custom coefficient bounds · force-include/exclude lists · configurable target · set-year window · due-tos by model year · execution-masked averages.
- **PowerPoint feedback (July):** the run-control restructure, batch grade labels, chart data labels, VIF column, Variables context columns, two-tier config, template up/download, Export tab, Definitions tab, final-model-on-all-data, and the Contributions time filter. All shipped; the Definitions screen and the user guide describe the behaviour.
- **Review (Alex + Arko, 2026-07-27):** the **fixed calendar window** and `InsufficientWindowData` (see Approach #5 — this fixed a real leak Alex found by pivoting the export in Excel) · contribution/due-to split · results cache · Hill saturation and the media-curve optimizer · `scale` column · Impact / SD · selection strictness presets and per-family caps.
- **Final update list (Alex, 2026-08-11):** the **High Level**, **Saturation** and **Multi-Level** screens · pooled and hierarchical models reproducing the sponsor's `Hierarchical Modeling Explanation.xlsx` to 1e-12 · his `SatCurve` class ported line-for-line · the interactive decay/midpoint/slope override he asked for on the call · and the **user guide** (`docs/`), the one item he called required rather than extra credit.

Two places where the implementation departs from the brief, both deliberate and both
documented in the code and the guide: national predictors are **multiplied** by each
channel's share rather than divided by it (the literal reading inflates contributions,
the opposite of the stated purpose — one line in `multilevel.proportionalize_national`),
and a **sign guard** clips the hierarchy index at 0, because on real data the raw index
runs −7.2 to +11.9 and a negative index silently undoes the sign priors set in the
variable config.

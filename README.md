# Big Chalk Mix Engine

**Automated marketing-mix regression engine for Brand × Retailer-Channel volume forecasting.**

A configuration-driven modeling system that estimates weekly Volume Sales — or any
chosen target — for every Brand × Channel combination in a client dataset, together
with a Plotly Dash application for running, steering, diagnosing and exporting those
models. The engine supports three levels of pooling (unpooled, pooled and
hierarchical), media adstock and saturation transforms, and constrained estimation
under analyst-supplied sign and coefficient bounds.

Developed as the Northwestern University MSDS Summer 2026 Capstone, sponsored by
**Big Chalk Analytics**.

| | |
|---|---|
| **Sponsor** | Big Chalk Analytics — Alex Hathcock (Senior Data Scientist), Arkaparna Sen |
| **Institution** | Northwestern University, MSDS Capstone, Summer 2026 |
| **Team** | Boqi Niu · Feifan Liu · Jiahao Li (Edison) |
| **Stack** | Python 3.10+ · NumPy · pandas · SciPy · statsmodels · Plotly Dash |
| **Status** | Feature complete; documentation delivered |

---

## Contents

1. [Confidentiality](#1-confidentiality)
2. [Overview](#2-overview)
3. [Installation](#3-installation)
4. [Usage](#4-usage)
5. [Application](#5-application)
6. [Methodology](#6-methodology)
7. [Validation](#7-validation)
8. [Results](#8-results)
9. [Project structure](#9-project-structure)
10. [Documentation](#10-documentation)
11. [Release history](#11-release-history)

---

## 1. Confidentiality

The client dataset (`Anonymized Data for Project.xlsx`), all presentation decks
(`*.pptx`) and all meeting transcripts are covered by the Big Chalk Analytics
non-disclosure agreement and are **excluded from this repository** by `.gitignore`.
Confidential material is held in the gitignored `local/` directory.

Development tooling files (`AGENTS.md`, `CODEX_HANDOFF.md`, `.agents/`, `.codex/`,
`.claude/`) are likewise excluded; they are not part of the deliverable.

The dataset must be obtained through the team's secure channel and placed in the
repository root before the engine is run.

---

## 2. Overview

Marketing-mix modeling attributes sales volume to the commercial levers a brand
controls — distribution, price, promotion, media — while adjusting for competitive
activity and macroeconomic conditions. Producing such a model for a single
brand-channel pair is routine. Producing and maintaining one for every brand in every
channel is not, and that is the problem this system addresses.

The engine builds 76 models in approximately 17 seconds, grades each one, and directs
the analyst's attention to the slices that require judgement rather than the ones that
do not.

### Capabilities

| Capability | Description |
|---|---|
| **Automated estimation** | VIF pruning and forward stepwise selection choose the model structure; a bounded least-squares fit enforces the analyst's sign and coefficient constraints. |
| **Configuration-driven framing** | Each variable's family, expected sign, bounds, adstock decay, saturation parameters and role are declared in a CSV. No variable names are hard-coded. |
| **Media transforms** | Normalized geometric adstock for carry-over and Hill saturation for diminishing returns, applied in a defined order. |
| **Multi-level modeling** | Unpooled, pooled and hierarchical estimation, scored on identical holdouts for direct comparison. |
| **Response curve analysis** | Execution-versus-ROI curves with average/marginal crossover, identifying optimal spend levels. |
| **Portfolio aggregation** | Roll-up of individual models to Total Brand or Total Channel views. |
| **Reproducible export** | Excel workbooks containing fit statistics, model structure and weekly decompositions, formatted for pivot-table analysis. |

---

## 3. Installation

**Requirements:** Python 3.10 or later.

```bash
git clone https://github.com/Jerry687/bigchalk-capstone.git
cd bigchalk-capstone
pip install -r requirements.txt
```

Place the client workbook in the repository root. The application detects worksheets,
channels and candidate target columns automatically.

---

## 4. Usage

### Application

```bash
cd code
python dashboard.py
```

The application starts on `http://127.0.0.1:8050`. A startup integrity check validates
every callback against the component layout and reports the result before serving.

### Command line

```bash
cd code
python run_brand1_channel1.py                     # Single slice, with EDA and diagnostics
python run_all.py                                 # Full batch → outputs/all_models_summary.csv
python run_all.py 1 3                             # Resumable batch, Brands 1–3
python multilevel.py "../Anonymized Data for Project.xlsx" "Brand 4"
                                                  # Three modeling levels for one product
```

### Verification

```bash
cd code
python test_multilevel.py            # 37 numerical assertions
python test_dashboard_screens.py     # 23 end-to-end interface assertions
python reference_alex_curvefit.py    # Equivalence against the sponsor's implementation
```

---

## 5. Application

The interface follows the "Bench" design system (IBM Plex, chalk-navy) delivered
through a Claude Design handoff. Nine screens are provided.

| Screen | Purpose |
|---|---|
| **Diagnostics** | Fit metrics, actual-versus-fitted with the out-of-sample forecast on the reserved tail, residual analysis, and a coefficient table with family classification, t-statistics and variance inflation factors. |
| **Variables** | The two-tier configuration editor: expected sign, adstock decay, scaling, saturation parameters, coefficient bounds and variable role, at product-default or product-channel-override scope. Supports bulk edit through template download and upload. |
| **Contributions** | Decomposition of modeled volume by driver and period, with the contribution/due-to distinction, period comparison and an uploadable fiscal-period mapping. |
| **High Level** | Aggregation of individual models to Total Brand, Total Channel or portfolio level, with overall fit, total contributions and year-over-year due-tos. |
| **Saturation** | Weekly comparison of raw, adstocked and saturated execution with correlations to the target; interactive override of decay, midpoint and slope; and the execution-versus-ROI response curve with its optimal spend point. |
| **Multi-Level** | Unpooled, pooled and hierarchical estimation for one product, scored on identical holdouts, with the national-predictor treatment, per-channel coefficient indices and the pooling-strength curve. |
| **Model Runs** | Batch execution across all Brand × Channel combinations, with model grades, review flags and the register of slices deliberately not modeled. |
| **Export** | Excel export for the current slice or the full portfolio. |
| **Definitions** | Reference documentation for every metric, threshold and method, maintained alongside the implementation. |

Run controls are confined to the Variables and Model Runs screens. All other screens
filter cached results and perform no estimation. Configuration writes are serialized
through a single process lock and committed transactionally.

---

## 6. Methodology

### 6.1 Leakage prevention

Decompositions of the target variable — `Volume Sales *`, `Dollar Sales *` — are
excluded from the predictor set dynamically, according to whichever target is selected.

### 6.2 Variable configuration

A two-tier configuration governs model framing. Each variable is assigned a family,
an expected sign, optional explicit coefficient bounds, a per-variable adstock decay,
optional saturation parameters, a reporting scale, and a role:

| Role | Behaviour |
|---|---|
| `auto` | Eligible for automated selection. |
| `force` | Retained in every model; exempt from VIF pruning and family caps. |
| `exclude` | Never considered. |

A **product default** applies to all channels; an optional **product × channel
override** takes precedence where present. A single resolver (`cp.resolve_config_path`)
is shared by the application, the batch runner and the single-slice runner, ensuring
all three agree.

### 6.3 Estimation

Model structure is selected by variance-inflation pruning followed by forward stepwise
entry. The reported model is then fitted by **bounded least squares**, enforcing sign
and coefficient constraints directly.

Ridge regression cannot honour sign bounds; stepwise selection is more transparent to a
client than Lasso; and variance-inflation analysis identifies multi-variable redundancy
that pairwise correlation does not. The fit has been verified numerically equivalent to
the sponsor's production `curve_fit` implementation to within 2 × 10⁻⁸ of the largest
coefficient's scale.

Selection strictness is adjustable through three presets (p < 0.01 / 0.05 / 0.15 with
corresponding VIF thresholds) and an optional per-family entry cap, applied during
selection rather than by post-hoc trimming.

### 6.4 Media transforms

Media variables are transformed in the order **raw → adstock → scale → saturation**.

**Adstock** models carry-over: `aₜ = (1 − decay)·xₜ + decay·aₜ₋₁`, normalized so that
adstocked totals approximate raw totals and never exceed them. Typical decays range from
0.2 for search to 0.7 for connected television.

**Hill saturation** models diminishing returns: `H(x) = xˢ / (xˢ + kˢ)`, where `k` is the
half-saturation point and `s` the curve shape. Adstock is applied first because
carry-over concerns when an impression lands; saturation last because diminishing
returns apply to accumulated pressure rather than to weekly spend. This ordering matches
Meta Robyn and Google Meridian.

**Scaling** divides a variable before fitting so that its coefficient may be read in
convenient units. This is a units transformation only: fitted values are identical to
within 1.2 × 10⁻¹⁰.

Saturation is disabled by default. The accompanying optimizer improves its own
cross-validation score considerably more reliably than it improves the untouched
holdout; supporting data is recorded in `outputs/curve_optimizer_experiment.csv`.

### 6.5 Modeling window and validation

The modeling window is an **absolute calendar range**, anchored to the latest week
present in the dataset and identical for every slice.

This is a correction of consequence. The window was previously the last *N* rows each
slice happened to possess, which meant a slice delisted in mid-2023 continued to produce
a model — estimated on 2023 data — that entered the weekly export as a disjoint segment
of history. Slices with insufficient data inside the window (fewer than 52 weeks, or
fewer than 26 weeks with non-zero sales) are **not modeled**, and are recorded with a
stated reason rather than omitted silently.

Within the window, a **validation model** selects structure on the weeks preceding a
reserved 13-week tail and is scored on that tail, yielding the reported out-of-sample
holdout MAPE. The **reported model** refits the same structure across the full window
and supplies the coefficients and decompositions presented. The in-sample fit is never
reported as a holdout result.

### 6.6 Contributions and due-tos

Two distinct quantities are maintained:

| Quantity | Definition | Question answered |
|---|---|---|
| **Contribution** | Volume a driver accounts for *within* a period. | "How much of my volume is media?" |
| **Due-to** | The change in a driver's contribution *between* two periods. | "Why is volume down year over year?" |

Decompositions are additive, with the intercept reported as its own line. Weekly
averages for media are computed only over weeks with actual execution. Periods of
unequal length are compared on a per-week basis.

Because coefficients on differently scaled variables cannot be ranked against one
another, an **Impact / SD** measure — coefficient multiplied by standard deviation — is
reported as the comparable quantity across drivers.

### 6.7 Multi-level modeling

Three levels of pooling are supported.

| Level | Specification |
|---|---|
| **Unpooled** | One independent model per Brand × Channel. Maximum flexibility; approximately 104 observations per model. |
| **Pooled** | One model per brand, with all channel-weeks stacked. One coefficient per predictor, estimated across 700–900 observations. |
| **Hierarchical** | Per-channel coefficients formed as the pooled coefficient multiplied by an index derived from that channel's unconstrained fit, subject to a variance cap and a shrinkage parameter. |

Stacking channels is admissible because these are models *of* a time series rather than
time-series models: adstock and saturation are applied within each channel before
stacking, leaving observations independent.

Pooled estimation requires two preparatory steps. **National predictors** — those
carrying identical values across all channels — are apportioned by each channel's share
of brand volume, so that the per-channel components sum to the national quantity and a
single shared coefficient does not multiply the effect by the channel count. **Channel
intercepts**, implemented as within-channel mean centering, allow channels differing by
a factor of thirty in scale to share slope coefficients without those coefficients
absorbing a level difference.

A shrinkage parameter λ interpolates between pooled (λ = 0) and fully indexed
hierarchical (λ = 1) estimation, and is fitted on the holdout.

### 6.8 Departures from specification

Two implementation decisions depart from the sponsor's written brief. Both are
documented in the source and in the user guide.

1. **National-predictor apportionment.** The brief specifies division by each channel's
   proportion of brand volume. Applied literally, this inflates contributions — the
   opposite of the stated objective. The implementation multiplies by the share, so that
   the per-channel components sum to the national quantity. The behaviour is isolated to
   `multilevel.proportionalize_national`.

2. **Sign guard on the hierarchy index.** The sponsor's specification contains no
   constraint on index sign. On the client data the unconstrained index ranges from
   −7.2 to +11.9, and a negative index inverts the sign of the final coefficient,
   silently overriding the sign priors declared in the variable configuration. The index
   is therefore clipped at zero.

---

## 7. Validation

| Suite | Coverage |
|---|---|
| `test_multilevel.py` | 37 assertions. Reproduces the sponsor's `Hierarchical Modeling Explanation.xlsx` column by column to within 10⁻¹²; verifies the `SatCurve` port against an independent Hill computation; confirms national-predictor conservation, exact decomposition, and continuity between pooling levels. |
| `test_dashboard_screens.py` | 23 assertions. Invokes every screen's callbacks directly and asserts a valid component tree rather than an error state. |
| `reference_alex_curvefit.py` | Equivalence against the sponsor's production `curve_fit` approach. |
| Startup integrity check | Validates every callback against the component layout at launch; the application refuses to start on a mismatch. |

Regression status: all 76 unpooled models are numerically identical to a direct engine
call, and the additive decomposition is exact to 4 × 10⁻¹⁵ across the portfolio.

---

## 8. Results

**Portfolio.** 76 models; median R² **0.881**; median holdout MAPE **11.5 %**. Six
slices are not modeled owing to insufficient data within the fixed window and are
enumerated with reasons in `outputs/all_models_skipped.csv`.

Slices with elevated error indicate structural change in the underlying business rather
than deficiency in the method. Brand 1 × Channel 1 lost distribution during the modeling
window; forcing `ACV Weighted Distribution` into the specification is what renders it
tractable, at a holdout MAPE of 19.6 %.

**Pooling.** No single level of pooling is universally preferable, which is itself the
principal finding. Across ten products, unpooled estimation is preferred for seven and
hierarchical for three; across the 76 individual channels, unpooled is preferred for 48,
hierarchical for 16 and pooled for 12. Pooling benefits thin channels that would
otherwise fit noise, and penalises the largest channel, which possesses sufficient data
to be estimated independently. The fitted shrinkage parameter reaches 1 for four
products and 0 for four others. Detail is recorded in `outputs/multilevel_comparison.csv`.

---

## 9. Project structure

```
code/
  capstone_pipeline.py          Core engine: loading, transforms, selection,
                                constrained estimation, contributions, configuration
  multilevel.py                 Pooled and hierarchical estimation
  saturation_curves.py          Response curves and weekly transform diagnostics
  rollup.py                     Portfolio aggregation
  dashboard.py                  Plotly Dash application
  assets/bench.css              Design system
  run_brand1_channel1.py        Single-slice runner
  run_all.py                    Batch runner
  reference_alex_curvefit.py    Sponsor equivalence check
  test_multilevel.py            Numerical test suite
  test_dashboard_screens.py     Interface test suite
configs/
  varconfig_<dataset>_<brand>.csv                  Product default
  varconfig_<dataset>_<brand>__ch_<channel>.csv    Product × channel override
outputs/
  all_models_summary.csv        Cross-slice fit summary
  all_models_skipped.csv        Slices not modeled, with reasons
  multilevel_comparison.csv     Pooling comparison by channel
  multilevel_summary_by_brand.csv
  curve_optimizer_experiment.csv
docs/
  BigChalk_Mix_Engine_User_Guide.docx / .pdf       Primary documentation
  build_user_guide.py                              Documentation build script
  Media_Curves_Math.md                             Transform derivations
  img/                                             Figures and interface captures
local/                          Confidential material (gitignored)
requirements.txt
```

Per-slice run output is regenerated on every batch execution and is not tracked.

---

## 10. Documentation

**`docs/BigChalk_Mix_Engine_User_Guide.docx`** is the primary deliverable: a 34-page
guide covering every screen and control, common workflows, metric definitions,
troubleshooting, and the reasoning behind the principal design decisions. It is
regenerated by `docs/build_user_guide.py`.

**`docs/Media_Curves_Math.md`** derives the adstock and Hill transforms.

The **Definitions** screen within the application provides the same reference material
adjacent to the figures it describes.

---

## 11. Release history

| Date | Scope |
|---|---|
| **2026-07-06** | Adstock normalization and per-media decays; configuration-table variable mapping; explicit coefficient bounds; force-include and exclude lists; configurable target; set-year window; due-tos by model year; execution-masked averages. |
| **July 2026** | Run-control restructure; batch grading; chart data labels; VIF reporting; two-tier configuration; template upload and download; Export and Definitions screens; final model fitted on all data; contribution period filtering. |
| **2026-07-27** | Fixed calendar window and insufficient-data handling (§6.5); contribution/due-to separation; results caching; Hill saturation and the curve optimizer; variable scaling; Impact / SD; selection strictness and family caps. |
| **2026-08-11** | High Level, Saturation and Multi-Level screens; pooled and hierarchical estimation; response curve implementation; interactive transform override; user guide. |

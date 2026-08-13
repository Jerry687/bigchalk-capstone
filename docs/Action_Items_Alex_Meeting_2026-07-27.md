# Action items — Alex + Arko review, 2026-07-27

Source: `local/Note_otter_ai_transcript.txt` (48 min, Alex Hathcock + Arko + team).

Overall tone: *"very cool, very impressed… this is already like meeting what I
thought would be at the end. Now we're just getting into the overly specific
requests and tweaks."* Four weeks left; the main deliverable should be closed
out, then extra credit.

Legend: **[DONE]** built and verified this session · **[NEXT]** extra credit,
scoped but not built.

---

## A. The data bug Alex found live

Alex pivoted the exported `weekly_due_tos` in Excel and hit a chart with a
chunk of data, then nothing, then a full series. Diagnosis on the call:
**Brand 1 and Brand 2 × Channel 8 have 25 weeks of data ending Jun 2023** —
18 months before the period being modeled — yet the engine produced models and
due-tos for them.

Root cause: the modeling window was **the last N rows each slice happened to
have**, not a date range. A delisted slice therefore got "its own" window in
2023, and the 13-week holdout landed on most of its data (hence Brand 1 ×
Channel 6's 254% MAPE).

> Alex: *"the window should be predefined, so the last 104 weeks should be the
> same 104 weeks for everything… instead of having the window being dependent
> on the data, the data is dependent on the window."*

**[DONE] Fixed window.** `cp.dataset_anchor_week()` finds the latest week in
the whole workbook (2025-12-28); the window is that anchor back N weeks and is
**identical for every Product × Channel** (104 wks → 2024-01-07 … 2025-12-28).
Verified: all 76 surviving models report the same window range.

**[DONE] Slices with no data in the window are not modeled.** New
`cp.InsufficientWindowData` — under 52 weeks of data or 26 selling weeks inside
the window, the slice is skipped with a stated reason. It shows in a
"Not modeled" table on Model Runs, in `all_models_skipped.csv` from the CLI,
and in the export's `errors` sheet. It never appears as a model.

Now skipped (was silently producing models):

| Slice | Why |
|---|---|
| Brand 1 × Channel 8 | 0 weeks in window (data ends Jun 2023) |
| Brand 2 × Channel 8 | 0 weeks in window (data ends Jun 2023) |
| Brand 1 × Channel 6 | 104 weeks, only **24** with non-zero sales ← the 254% MAPE model |
| Brand 2 × Channel 5 | 104 weeks, only **11** with non-zero sales |
| Brand 5 / Brand 7 × Channel 6 | not sold in channel |

Side effect, also verified: the exported weekly due-tos now span exactly
2024-01-07 → 2025-12-28 with no disjoint pre-window chunk.

---

## B. Contribution vs due-to — a terminology error in the tool

Arko: *"due-to is like a change, right?"* Alex confirmed and corrected himself:
*"what you have here is contribution — what is the total per period. The due-to
is just the change… that's really what all the people that look at this
actually care about: how is the change of macroeconomic variables impacting my
sales year to year."*

**[DONE]** Both are now computed and clearly separated:

- **Contribution** = level in a period (Σ βᵢ·xᵢ over the period's weeks).
- **Due-to** = contribution in period B − contribution in period A.

A toggle on Contributions switches the driver chart between them. Periods of
unequal length are compared as **per-week averages** automatically (otherwise a
52-week period beats a 4-week one for reasons unrelated to the drivers).
Engine helpers `cp.contributions_by_period` / `cp.due_to_change` back it.
Verified: due-tos sum to the change in modeled volume to 0.000000.

Alex also asked to keep the totals and averages — *"it's still fine to have the
total due-to and the average"* — those views are unchanged.

---

## C. Re-running on every filter change

> Alex: *"instead of having to rerun every time, you just create the data that
> filters into this for everything, and then you just filter down… it's going
> to take a little bit longer for the first one, but then when we're
> evaluating, it's instantaneous."*

He also flagged the real hazard: *"I suddenly hit window to 156 weeks and run,
and now I've got one model 156, everything else 104."*

**[DONE] Results cache.** A Model Runs batch computes every slice once (~19s
for 76 models) and caches the full results, keyed by (datafile, target,
window). Diagnostics / Contributions / Export read and filter that cache —
changing Product or Channel refits nothing. Re-tuning one slice on Variables
and hitting Run refreshes just that entry. If a slice isn't in the cache the
screen says why (never the previous slice's numbers).

*Note for Alex:* he asked us to report back if generating everything up front
turned out slower than re-running one at a time. It isn't — a single slice is
~0.25s and the full batch ~19s, so the cache pays for itself after roughly
80 filter changes, and the consistency guarantee is worth more than that.

---

## D. Modeling controls on results screens

> Alex: *"remove everywhere that you can change the target. You should only be
> able to change the target in one spot… I don't think you should have any
> targets here. The only thing you should be filtering on are results, which
> would be your product and channel."*

**[DONE]** TARGET and WINDOW are off the top bar and live in one place — a
"Modeling settings" block on Variables. The top bar keeps PRODUCT + CHANNEL
(filters) and echoes the settings read-only with the actual date range, e.g.
`Volume Sales · 104 wks · Jan 07, 2024 – Dec 28, 2025`. This satisfies his
"if you want to note somewhere, like these are volume sales due-tos on 140 mil,
that's fine, but it shouldn't be a filter or dropdown."

Run buttons remain only on Variables (tuning) and Model Runs (batch).

---

## E. Export additions

> *"I would add just one more thing here… add another column to the side that
> says what the coefficient currently is."* · *"maybe add a flag for what made
> it through the model."* · *"if we have uploaded a time map, include that in
> here."*

**[DONE]** `fit_structure` sheet gains `current_coefficient` and
`in_last_model` (both ignored on re-upload, so the round-trip still works).
`weekly_due_tos` gains `model_year`, `period` (from the uploaded time map),
`quarter` and `year` columns so it pivots without a VLOOKUP — this is exactly
the pivot Alex walked the team through on screen. New `due_to_change` sheet
carries the period-over-period due-tos. Fit statistics gains the window range
and the weeks/selling-weeks counts.

**[DONE] Chart labels.** *"push the labels to the left so they're not
intervening with the table… Competition ACV Weighted Distribution is kind of
cutting off the number at the top graph."* — y-axis `automargin` plus 18% x
padding for the outside value labels.

---

## F. Extra credit

Alex: *"let's get this as done as possible because this is the main one, and
then we're going to get into extra credit."*

**All four are built.** F1 (scaling / comparable coefficients), F4 (selection
hyperparameters + per-family caps), and F2+F3 (adstock optimization +
Hill saturation curves, with a response-curve chart).

Everything here defaults to OFF or to the previous behaviour, so the 76
baseline models are byte-for-byte unchanged until someone opts in — verified.

### F1. [DONE] Scaling and comparable coefficients (Arko's ask)

Arko: coefficients aren't comparable because the inputs aren't on comparable
supports — *"personal savings rate is a percentage… versus a media variable
[with] impressions… in the millions."* Alex: put a **scale factor in the
variable config**, and look at **normalizing** inputs.

**What we built, and why not literal normalization.** For an unregularized
constrained least-squares fit, min-max normalization is a linear
re-parameterization — Arko said so himself: *"it does nothing to your model."*
Our selection step already z-scores internally, so variable selection was
already scale-invariant. Normalizing and then inverting every reported number
would therefore add the exact risk Arko warned about (*"you have to go back to
the original scale"*) while changing no result. So we solved the actual
problem — comparability — directly:

- **`scale` column in the variable config.** Divides the variable before
  modeling, so a media coefficient reads *per 1,000 impressions*. Custom
  bounds are written in original units and transformed with the scale, so
  setting a scale can't silently tighten a bound. Verified a units change
  only: fitted values identical to 1.2e-10, same variables selected,
  contributions unchanged.
- **`Impact / SD` column** (β × SD) on the coefficient table, the Variables
  tab and the export — volume moved by a one-SD change, the same unit for
  every driver. This is the column that can legitimately be ranked. On
  Brand 1 × Channel 1 it makes the point: raw coefficients of 61,473
  (seasonality) vs 3,107 (ACV) look 20× apart; their impacts are 13,193 vs
  13,151 — effectively tied. `β × range` is also computed.

Both are invariant to `scale`, so the comparison is stable however the analyst
sets the units.

*If Arko still wants literal normalization*, the remaining piece is fitting on
normalized inputs and inverting the coefficients — worth doing only for
numerical conditioning, and it should come with a test asserting it reproduces
the current fit.

### F2 + F3. [DONE] Adstock optimization + saturation curves

Built together — they share one optimization engine. Full math in
`docs/Media_Curves_Math.md`.

**The curve is the Hill equation.** Alex: *"some Bill and this other data
scientist created this for the efficacy of drugs… a curve that has a midpoint
and a slope."* That is **A. V. Hill**, 1910, oxygen binding to haemoglobin,
and since then the standard dose–response curve in pharmacology — he had the
name slightly off. It is also the saturation function in Meta's Robyn and
Google's Meridian. Two parameters, a midpoint and a slope, exactly as he
described. So we did not need to wait for his notes.

    H(x) = xˢ / (xˢ + kˢ),   k = midpoint × peak execution

Transform order is raw → adstock → scale → Hill, as in Robyn/Meridian.
Because H lands in [0,1), the coefficient on a saturated variable is the
**maximum weekly volume that variable can deliver**. Contributions stay
additive, so the due-to decomposition is unchanged (verified to 1e-10).

**The optimizer** searches decay × midpoint × slope per media variable by
coordinate descent. The design decision that matters is what it is scored on:
the window is split THREE ways — train / rolling inner folds / **outer
holdout that the search never touches** — so the holdout the dashboard reports
afterwards is still an honest number. A candidate is adopted only if it is not
worse in *any* fold and improves the mean by ≥0.5 pp.

**What we found — and this is the part to present.** Across 25 slices
(`outputs/curve_optimizer_experiment.csv`):

| | result |
|---|---|
| cross-validated MAPE (what it optimizes) | improved **4 of 4** times it fired, mean −2.07 pp |
| untouched holdout | improved **2 of 4**, mean **+0.85 pp** |

It reliably improves the thing it optimizes; transfer to genuinely unseen data
is roughly a coin flip. It also **declines to change anything on 10 of the 14
slices** that have media — the conservative criteria are working. Best case
was Brand 4 × Channel 2 (three media variables): holdout 9.84% → 6.90%. Worst
was Brand 1 × Channel 1: 19.64% → 25.51%.

Why: media is a small share of these models, and 104 weeks is thin for two
extra non-linear parameters per media variable. Robyn and Meridian handle this
with Bayesian priors and pooling across geographies — the priors do the work
the data can't. Partial pooling across a brand's channels is the natural next
step.

**So it ships off by default**, as an opt-in tool on Variables that writes
*candidate* curves into the config and reports its CV gain. It does not save
and does not run — the analyst runs it and checks the holdout. An engine that
tells you when its own optimization didn't generalize is worth more than one
that always claims a win.

**Also built: media response curves** (Diagnostics). Fitted curve per media
variable with the weeks actually executed plotted on it, and the
half-saturation point marked. Where the dots sit is the operating point —
clustered on the steep part means headroom, out on the flat means extra spend
is buying little. That is Alex's *"you start seeing diminishing returns"* made
visible.

Runtime: ~1.7 s per slice, ~4 minutes for all 76.

### F4. [DONE] Variable-selection hyperparameters, incl. per-group caps

> *"is there a way to have variable hyperparameter tuning — we want to be more
> restrictive, we want to be less restrictive… more restrictive by a group.
> I want only two to three trade, but I want all media."*

Built on the Variables screen, next to the modeling settings:

- **Strictness**: Punitive (p<0.01, VIF>5) · Balanced (p<0.05, VIF>10) · Lax
  (p<0.15, VIF>20). Balanced is the existing default, so nothing moves until
  someone opts in. Measured across all 76 slices, average predictors per
  model: **4.4 / 6.5 / 10.0** — the knob does what it says.
- **Max variables per family**: an editable table, blank = no limit. Alex's
  example is Trade 3, Competitive 2, Macro 2, Media blank. Enforced *during*
  selection rather than by trimming afterwards, so a capped family spends its
  budget on its most significant members and the other families still fill
  normally. Forced variables are always kept and count toward the budget.

Defaults are **no caps**, so all 76 existing models are unchanged (verified).
The results cache is keyed by these hyperparameters as well as target and
window, so you can never see a blend of two settings on one screen — change a
knob and the screens ask for a fresh batch.

Worth showing Alex: on Brand 1 × Channel 1, capping Trade at 1 and Macro at 1
*improved* holdout MAPE from 23.0% to 21.5% — his instinct that fewer, better
variables per group can beat an unconstrained search has some support.

---

## G. Still open from 2026-07-21

- Runtime / scaling email: time 1 / 10 / full-set / doubled runs, is it linear.
  (`code/scaling_test.py` exists; the batch is now ~19s for 76 models, so the
  numbers need a refresh under the new cache.)

## H. Soft feedback

- Learn Excel. *"You're programming off the charts… but no one's teaching Excel
  anymore. I'd recommend learning Excel."* Specifically pivot tables — that's
  how the client will interrogate these exports, which is why the period
  columns in E matter.
- Think like the end user: *"think about how you would use this if you were
  running 1000 models. What would make your life easier?"*

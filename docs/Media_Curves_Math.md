# Media response curves — the math

For Alex and Arko. Covers the two non-linearities of media and the engine that
fits them. Written so it can be read without the code.

---

## 1. Where this came from

Alex, 2026-07-27:

> *"That's part one of what we would call the nonlinearity of the effect of
> media on consumption… There is a second level of nonlinearity called a
> saturation curve, and basically what we do is some Bill and this other data
> scientist created this for the efficacy of drugs… it is a curve that has a
> midpoint and a slope. So it makes it sigmoidal."*

**The name is Hill, not Bill.** Archibald Vivian Hill, 1910, fitting the
binding of oxygen to haemoglobin; the equation became the standard
dose–response curve in pharmacology, which is exactly the "efficacy of drugs"
Alex remembered. It is also the saturation function in Meta's Robyn and
Google's Meridian, the two open-source MMM stacks. Two parameters: a midpoint
and a slope — precisely what he described. So we did not need to wait for his
notes; this is the curve, and it is the industry standard for the job.

---

## 2. Transform order

    raw execution  →  adstock (carry-over)  →  scale (units)  →  Hill (saturation)

Adstock first because carry-over is about *when* an impression lands.
Saturation last because diminishing returns apply to the accumulated
advertising pressure in a week, not to each raw dollar independently. Same
order as Robyn and Meridian.

## 3. Adstock — carry-over (already in the engine since July 6)

    a_t = (1 − d)·x_t + d·a_{t−1}

`d` is the decay. The (1−d) share of this week's execution lands this week; the
rest carries forward. Normalized so the adstocked total ≈ the raw total — the
naive `a_t = x_t + d·a_{t−1}` inflates total impressions, which then inflates
the contribution.

Alex's framing: *"if you see an ad and you wait to do grocery shopping until
next week, it's going to have a decayed amount."* Search ≈ 0.2 (acted on
immediately), TV/CTV ≈ 0.7 (long memory), 0.5 default.

## 4. Hill — saturation

    H(x) = x^s / (x^s + k^s),        k = γ · ref

- **`k` — the midpoint.** At `x = k` the response is exactly half its maximum.
  Stored as a fraction `γ ∈ (0,1]` of a reference scale `ref` (the peak
  adstocked execution) so the parameter is comparable across brands and
  channels whose spend differs by orders of magnitude.
- **`s` — the slope.** `s > 1` gives the S-curve Alex described: slow start
  (*"the first 10 impressions probably don't do a lot"*), steep through the
  midpoint, then flattening. `s < 1` gives a curve that diminishes straight
  from the origin, which weekly aggregate data often prefers.

`H` returns a value in [0,1), so on a saturated variable the **coefficient is
the maximum weekly volume that variable can deliver** — a genuinely useful
number in its own right. Contributions stay additive, so
`intercept + Σ βᵢ·H(xᵢ)` still reproduces the fitted line exactly and every
due-to still decomposes (verified to 1e-10).

**`ref` is computed on training weeks only.** If it came from the whole window
the transform itself would have seen the holdout — a subtle leak that would
have flattered every out-of-sample number we report.

---

## 5. The optimizer

Alex: *"Can we create an optimization engine that will pick the best ad stock
based on correlation with the data, predictive power in a regression? … Is
there a certain slope and midpoint that best fits the data?"*

Search space per media variable: decay `d` × midpoint `γ` × slope `s`, plus
the option of no saturation at all (so the search can always decline).

    d ∈ {0, 0.2, 0.4, 0.6, 0.8}
    γ ∈ {0.2, 0.35, 0.5, 0.65, 0.8}
    s ∈ {0.5, 1.0, 1.5, 2.5}

**Coordinate descent**, one variable at a time, two passes. A full joint search
is grid^k in the number of media variables; coordinate descent is k×grid per
pass and finds the same optimum on a near-separable problem like this.

### What it is scored on — the part that matters

Tuning curve shapes is a search over ~105 combinations per variable. Whatever
you score on **will** be over-fitted; that is arithmetic, not bad luck. So the
window is split three ways:

    [ ───────── train ───────── | inner folds | OUTER HOLDOUT ]
                                                   13 wks

- Candidates are scored by **rolling-origin cross-validation** on the inner
  folds: train on everything before a fold, score the fold, average over three
  expanding-window folds. A curve must help in several different periods, not
  one lucky quarter.
- A candidate is adopted only if it is **not worse in any single fold**
  (strict dominance) and improves the mean by at least 0.5 pp.
- The **outer holdout is never touched during the search.** So the holdout
  MAPE the dashboard reports afterwards remains an honest out-of-sample number,
  directly comparable with a model that was never optimized.

That last point is the whole design. Had we scored the optimizer on the
reported holdout, every optimized model would have looked better and not one
of them would have been.

---

## 6. What we found — read this before using it

We ran the optimizer over 25 slices (14 of which have media in the model) and
compared the cross-validated score it optimizes against the untouched holdout.
Full results: `outputs/curve_optimizer_experiment.csv`.

**The optimizer reliably improves the thing it optimizes. Whether that
transfers to the untouched holdout is close to a coin flip.**

| | slices | mean change |
|---|---|---|
| cross-validated MAPE (what it optimizes) | improved **4 of 4** | **−2.07 pp** |
| reported holdout (never touched) | improved **2 of 4** | **+0.85 pp** (median +0.24) |

Per slice:

| slice | media | curves changed | CV before → after | holdout before → after |
|---|---|---|---|---|
| Brand 1 × Channel 1 | 2 | 1 | 12.37 → 11.57 | 19.64 → **25.51** |
| Brand 1 × Channel 2 | 1 | 1 | 19.60 → 18.22 | 7.22 → 7.13 |
| Brand 3 × Channel 2 | 1 | 1 | 7.88 → 4.91 | 6.07 → 6.65 |
| Brand 4 × Channel 2 | 3 | 2 | 7.06 → 3.90 | 9.84 → **6.90** |

Two things worth noting in the optimizer's favour:

- **It declines far more often than it fires.** Of the 14 slices with media, it
  changed curves on only 4. The strict-dominance rule and the 0.5 pp floor
  reject most candidates — the search is not grabbing at every apparent gain.
- **When it fires on a slice with real media weight it can be worth it.**
  Brand 4 × Channel 2 has three media variables and gained 2.94 pp of genuine
  out-of-sample accuracy. Brand 1 × Channel 1 has two, and lost 5.87.

An earlier version scored on a single 13-week inner window instead of rolling
folds, and was clearly worse — it took Brand 1 × Channel 1 from 19.64% to
21.87% while claiming an inner-window gain. Rolling-origin folds plus strict
dominance improved the discipline but did not make the transfer reliable.

Runtime: ~1.7 s per slice on average (up to ~9 s for a slice with three media
variables), so optimizing all 76 slices is roughly a 4-minute job.

Two reasons, both worth saying out loud to the client:

1. **Media is a small share of these models.** The dominant drivers are
   distribution, seasonality, trade and competitive pressure. Refining the
   shape of a small term cannot buy much accuracy, but it can add variance.
2. **104 weeks with ~10 predictors is not much data to identify two extra
   non-linear parameters per media variable.** Robyn and Meridian fit these
   with Bayesian priors and hierarchical pooling across geographies for exactly
   this reason — the priors do the work the data cannot.

### Therefore

- Saturation and decay optimization ship **off by default**. Every existing
  model is unchanged.
- The optimizer is a **tool on the Variables screen**: it writes candidate
  curves into the config table and reports its cross-validated gain. It does
  not save and does not run. The analyst reviews, runs, and sees what the
  curves did to the honest holdout.
- If the holdout gets worse, the curves were fitting noise. Revert them.

This is a real finding, not a failure: an automated engine that reports when
its own optimization does not generalize is more useful than one that always
claims an improvement.

### Where it should help more

- Slices with heavy, sustained media where a saturation region is actually
  observed in the data.
- A longer window (156 weeks) — more weeks per non-linear parameter.
- Pooling curve parameters across channels within a brand (partial pooling),
  so one channel's thin media history borrows strength from the others. This
  is the natural next step and is how the commercial tools solve it.

---

## 7. Reading the response-curve chart

Diagnostics → *Media response curves*. Each line is a fitted curve; the dots
are the weeks that variable actually ran.

- **Flattening curve** — diminishing returns. The dotted vertical is the
  half-saturation point: past it, the next unit of execution returns less than
  the last.
- **Straight line** — no saturation was fitted; the variable is linear in
  execution.
- **Where the dots sit** is the operating point. Dots clustered on the steep
  part say there is headroom; dots out on the flat say the channel is
  saturated at current levels and extra spend is buying little. That is the
  read Alex wanted: *"you start seeing diminishing returns with running your
  media."*

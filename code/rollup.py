"""
TOTAL VIEW — rolling Brand × Channel models up to Total Brand or Total Channel
═══════════════════════════════════════════════════════════════════════════════
Alex, "Last Set of Dashboard Updates" (2026-08-11):

    "High Level View: Add a 'Total' view for Brand or Channel. This means that
     each brand*channel is rolled up to either Total Brand or Total Channel, to
     see the overall fit and contributions."

WHAT IS AND IS NOT BEING COMPUTED HERE
──────────────────────────────────────
This aggregates RESULTS, it does not fit a new model. Every Brand × Channel
model keeps its own coefficients; what is summed is the OUTPUT — actual volume,
fitted volume, and each driver's weekly contribution — across the slices in the
group. Because the engine's decomposition is additive and exact within each
slice (intercept + Σβx == fitted, verified to 1e-10), it stays exact after
summation: the rolled-up contributions still add to the rolled-up fitted line.

That distinction matters when reading the fit statistics. Rolled-up R² and MAPE
describe how well the PORTFOLIO of models tracks total volume — which is the
question an account lead actually asks ("does this thing predict my business?")
— and they are systematically kinder than the individual models, because
independent errors partially cancel when you add series together. That is a
real property of the aggregate, not a trick, but it is why the per-slice grid is
shown alongside: a total that fits well can still hide a channel that does not.
A user who wants "one model for the whole brand" wants the POOLED model
(`multilevel.run_pooled`), which is a different thing and is offered separately.

Two groupings, per Alex's "either Total Brand or Total Channel":
  Total Brand    — for each brand, sum across its channels  ("how is Brand 1
                   doing overall, across every channel it sells in")
  Total Channel  — for each channel, sum across brands      ("how is Channel 3
                   doing overall, across every brand we model in it")
And "Total" collapses everything into one line for the whole dataset.

Alignment is by WEEK, not by position: every slice now shares one fixed
calendar window (`cp.dataset_anchor_week`, the fix from 2026-07-27), so dates
line up — but summing by date rather than by row index means a slice that is
missing a week contributes nothing to that week instead of silently shifting
its whole series by one, which is the kind of error that is invisible in a
chart and fatal in an export.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _slice_frame(p: dict) -> pd.DataFrame:
    """One payload's weekly actual / fitted / per-driver contributions, indexed
    by date."""
    cw = p.get("contrib_wk") or {}
    dates = pd.to_datetime(cw.get("dates") or p["dates"])
    out = pd.DataFrame({"actual": p["actual"], "fitted": p["fitted"]},
                       index=dates)
    for driver, vals in (cw.get("contrib") or {}).items():
        out[f"c::{driver}"] = vals
    return out


def rollup(payloads: list, group_by: str = "brand") -> dict:
    """Aggregate a list of slice payloads into Total views.

    Parameters
    ----------
    payloads : the per-slice dicts the dashboard already caches (`_payload`).
    group_by : "brand"   -> one total per brand, summed over its channels
               "channel" -> one total per channel, summed over its brands
               "all"     -> a single total over everything

    Returns {"groups": {label: {...}}, "group_by": ..., "n_slices": ...}
    where each group carries weekly actual/fitted, per-driver contributions,
    fit statistics recomputed on the aggregate, and the member slices.
    """
    if group_by not in ("brand", "channel", "all"):
        raise ValueError("group_by must be 'brand', 'channel' or 'all'")

    buckets: dict = {}
    for p in payloads:
        key = ("Total" if group_by == "all"
               else f"Total {p['brand']}" if group_by == "brand"
               else f"Total {p['channel']}")
        buckets.setdefault(key, []).append(p)

    groups = {}
    for label, members in buckets.items():
        frames = [_slice_frame(p) for p in members]
        # ── WHY concat + groupby AND NOT A CHAIN OF DataFrame.add ─────────
        # `a.add(b, fill_value=0)` fills only where ONE side is missing. Two
        # slices can differ in BOTH columns and dates — a driver selected in
        # one channel but not another, and (in this data) a slice whose window
        # covers a slightly different set of weeks. Where a column is new AND
        # the date is one the running total has not seen, both sides are
        # missing, fill_value does not apply, and the cell comes out NaN,
        # silently poisoning that driver's whole column. It showed up on Brand
        # 9 / Channel 6, on a competitor variable no other slice had selected.
        # Stacking every slice and summing by date makes "this slice did not
        # use this driver" mean zero, which is what it means.
        stacked = pd.concat(frames, axis=0)
        coverage = stacked.groupby(level=0).size()
        agg = stacked.fillna(0.0).groupby(level=0).sum().sort_index()

        actual = agg["actual"].values.astype(float)
        fitted = agg["fitted"].values.astype(float)
        resid = actual - fitted
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        nz = actual != 0
        denom = float(np.abs(actual).sum())

        driver_cols = [c for c in agg.columns if c.startswith("c::")]
        contrib = {c[3:]: [float(v) for v in agg[c].values] for c in driver_cols}
        # exactness carries over from the per-slice decomposition
        recon = np.sum([agg[c].values for c in driver_cols], axis=0) \
            if driver_cols else np.zeros_like(fitted)
        scale = max(float(np.abs(fitted).max()), 1.0)

        totals = {k: float(np.sum(v)) for k, v in contrib.items()}
        drivers = sorted([k for k in totals if k != "Intercept"],
                         key=lambda k: abs(totals[k]), reverse=True)

        groups[label] = {
            "label": label,
            "members": [{"brand": p["brand"], "channel": p["channel"],
                         "r2": p["stats"]["r2"], "mape": p["stats"]["mape"],
                         "holdout_mape": p["stats"]["holdout_mape"],
                         "volume": float(np.sum(p["actual"])),
                         "n_vars": p["stats"]["n_selected"]}
                        for p in sorted(members,
                                        key=lambda q: -float(np.sum(q["actual"])))],
            "n_slices": len(members),
            "dates": [d.strftime("%Y-%m-%d") for d in agg.index],
            "actual": [float(v) for v in actual],
            "fitted": [float(v) for v in fitted],
            "resid": [float(v) for v in resid],
            "contrib": contrib,
            "drivers": drivers,
            "totals": totals,
            "stats": {
                "r2": (1 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot else np.nan,
                # MAPE on the aggregate series, and WMAPE — for a total, WMAPE
                # is the number that means "how far off is my volume", because
                # it is error dollars over actual dollars rather than an
                # average of week-level percentages
                "mape": (float(np.mean(np.abs(resid[nz] / actual[nz]))) * 100
                         if nz.any() else np.nan),
                "wmape": (float(np.abs(resid).sum()) / denom * 100
                          if denom else np.nan),
                "volume": float(actual.sum()),
                "fitted_volume": float(fitted.sum()),
                "bias_pct": (float(resid.sum() / actual.sum()) * 100
                             if actual.sum() else np.nan),
                "n_slices": len(members),
                "n_weeks": len(agg),
                # A week that only some slices cover is a partial total and
                # will dip for a reason that has nothing to do with the
                # business. Reported so the chart can be read correctly rather
                # than left to look like a real decline.
                "weeks_full_coverage": int((coverage == len(members)).sum()),
                "min_slices_in_a_week": int(coverage.min()),
                # every member's worst week, so a good total can't hide a bad
                # slice without saying so
                "worst_member_mape": max(
                    (m["mape"] for m in
                     [{"mape": p["stats"]["mape"]} for p in members]),
                    default=np.nan),
                "decomposition_error": float(np.abs(recon - fitted).max()) / scale,
            },
        }
    return {"groups": groups, "group_by": group_by,
            "n_slices": sum(len(v) for v in buckets.values())}


def rollup_table(rolled: dict) -> pd.DataFrame:
    """One row per Total group — the summary grid for the High Level screen."""
    rows = []
    for label, g in rolled["groups"].items():
        s = g["stats"]
        rows.append({
            "group": label, "slices": s["n_slices"], "weeks": s["n_weeks"],
            "volume": s["volume"], "r2": s["r2"], "mape": s["mape"],
            "wmape": s["wmape"], "bias_pct": s["bias_pct"],
            "worst_slice_mape": s["worst_member_mape"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("volume", ascending=False).reset_index(drop=True) \
        if len(out) else out


def contribution_table(group: dict, top: Optional[int] = None) -> pd.DataFrame:
    """Total contribution per driver for one rolled-up group, with each
    driver's share of modeled volume — the "what drove the business" table at
    the total level."""
    tot = group["totals"]
    base = float(sum(tot.values())) or np.nan
    rows = [{"driver": d, "contribution": float(tot[d]),
             "share_pct": float(tot[d]) / base * 100 if base == base else np.nan}
            for d in ["Intercept"] + group["drivers"] if d in tot]
    out = pd.DataFrame(rows)
    return out.head(top) if top else out


def rollup_due_to(rolled: dict, label: str, weeks_a: int = 52,
                  weeks_b: int = 52) -> pd.DataFrame:
    """Year-over-year due-to at the rolled-up level: the LAST `weeks_b` weeks
    against the `weeks_a` weeks before them.

    Due-to, not contribution — the distinction Arko drew on 2026-07-27 and the
    one the engine already respects everywhere else. This is the CHANGE each
    driver explains between the two periods, which is what "why is Total Brand 1
    down this year" actually asks.
    """
    g = rolled["groups"][label]
    n = len(g["dates"])
    b0, a0 = max(n - weeks_b, 0), max(n - weeks_b - weeks_a, 0)
    rows = []
    for d, vals in g["contrib"].items():
        v = np.asarray(vals, dtype=float)
        a, b = float(v[a0:b0].sum()), float(v[b0:].sum())
        na, nb = max(b0 - a0, 1), max(n - b0, 1)
        if weeks_a != weeks_b or na != nb:      # compare per-week if uneven
            a, b = a / na * min(na, nb), b / nb * min(na, nb)
        rows.append({"driver": d, "prior": a, "latest": b, "due_to": b - a})
    out = pd.DataFrame(rows)
    total = float(out["due_to"].sum())
    out["share_of_change_pct"] = (out["due_to"] / total * 100
                                  if total else np.nan)
    return out.sort_values("due_to", key=lambda s: s.abs(), ascending=False) \
        .reset_index(drop=True)

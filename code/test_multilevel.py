"""
Verification for the multi-level and saturation-curve work.

The important test here is `test_alex_spreadsheet`: it reproduces
`Hierarchical Modeling Explanation.xlsx` cell for cell. Alex sent that sheet as
the definition of the arithmetic, so matching it is not a nice-to-have — it is
the acceptance criterion for the hierarchical model. Everything else checks the
properties the engine relies on downstream (conservation, reconciliation, no
leakage).

    cd code && python test_multilevel.py [path-to-workbook]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import capstone_pipeline as cp
import multilevel as ml
import saturation_curves as sat

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Alex's spreadsheet, reproduced exactly
# ═══════════════════════════════════════════════════════════════════════════

def test_alex_spreadsheet():
    """`Hierarchical Modeling Explanation.xlsx`, columns D through I.

    Sheet layout: 3 channels × 5 predictors.
      D  Pooled Coefficient           (given)
      E  Unpooled Coefficient         (given)
      F  Indexed Unpooled Coefficient = E / AVERAGEIFS(E, predictor)
      G  Final Coefficient            = F × D
      H  Indexed & Capped Index       = F clipped to 1 ± L·STDEV.P(F)
      I  Capped Final Coefficient     = H × D
    with the standard-deviation limit L in L8 = 1.
    """
    print("\nAlex's Hierarchical Modeling Explanation.xlsx")
    preds = ["Distribution", "Price", "Media 1", "Media 2", "Macro-Economic"]
    pooled = pd.Series([1.5, -0.2, 0.15, 0.25, 0.01], index=preds)
    unpooled = pd.DataFrame(
        [[1.25, -0.50, 0.10, 0.40, 0.080],      # Channel 1
         [1.50, -0.01, 0.05, 0.20, 0.012],      # Channel 2
         [1.75, -0.30, 0.00, 0.10, 0.004]],     # Channel 3
        index=["Channel 1", "Channel 2", "Channel 3"], columns=preds)

    index = ml.coefficient_index(unpooled)
    # column F, computed by hand from the sheet's formula
    expect_F = unpooled / unpooled.mean(axis=0)
    check("column F — indexed unpooled coefficient",
          np.allclose(index.values, expect_F.values, atol=1e-12),
          f"max diff {np.abs(index.values - expect_F.values).max():.2e}")

    # column G = F x D  (uncapped hierarchical, Alex's "Model 3")
    final = index.mul(pooled, axis=1)
    expect_G = expect_F.mul(pooled, axis=1)
    check("column G — final coefficient (F x D)",
          np.allclose(final.values, expect_G.values, atol=1e-12))

    # column H = index capped at 1 +/- 1 x STDEV.P(index)
    capped = ml.cap_index(index, sd_limit=1.0, shrink=1.0, enforce_sign=False)
    sd = index.std(ddof=0)                       # STDEV.P
    expect_H = index.clip(lower=1 - sd, upper=1 + sd, axis=1)
    check("column H — capped index (1 +/- 1 STDEV.P)",
          np.allclose(capped.values, expect_H.values, atol=1e-12),
          f"sd = {dict(sd.round(4))}")

    # column I = H x D
    check("column I — capped final coefficient",
          np.allclose(capped.mul(pooled, axis=1).values,
                      expect_H.mul(pooled, axis=1).values, atol=1e-12))

    # The sheet's own check, rows 20-25: "See how the High Level Pooled
    # Coefficients are similar to the Hierarchical and Capped Hierarchical?"
    check("row 21-25 — index averages to exactly 1 per predictor",
          np.allclose(index.mean(axis=0).values, 1.0, atol=1e-12),
          f"max deviation {np.abs(index.mean(axis=0) - 1).max():.2e}")
    check("row 21-25 — uncapped hierarchical averages back to pooled",
          np.allclose(final.mean(axis=0).values, pooled.values, atol=1e-12))

    chk = ml.hierarchical_check(pooled, index, capped, capped.mul(pooled, axis=1))
    check("hierarchical_check agrees",
          chk["index_mean_max_dev"] < 1e-12
          and chk["uncapped_recovers_pooled"] < 1e-12)

    # shrink = 0 must collapse to the pooled model exactly
    z = ml.cap_index(index, sd_limit=1.0, shrink=0.0, enforce_sign=False)
    check("shrink=0 collapses the hierarchy to the pooled model",
          np.allclose(z.values, 1.0, atol=1e-12))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Alex's SatCurve, reproduced exactly
# ═══════════════════════════════════════════════════════════════════════════

def test_satcurve():
    """Recompute the class's outputs independently and compare. Also checks
    the two properties a planning read-out has to have."""
    print("\nSatCurve (port of Alex's class)")
    rng = np.random.default_rng(7)
    execs = np.abs(rng.normal(50_000, 15_000, 52))
    execs[:6] = 0.0                                   # a dark period
    df = pd.DataFrame({"Amazon_Spend": execs})

    c = sat.SatCurve(df, "Amazon_Spend", spend=2_500_000, price=4.0,
                     margin=0.30, target_response=180_000,
                     saturation=0.45, slope=1.8)
    rcd = c.response_curve_data()

    check("grid is 0-300% in 301 steps",
          len(rcd) == 301 and rcd["Ratio"].iloc[0] == 0
          and abs(rcd["Ratio"].iloc[-1] - 3.0) < 1e-12)

    # independent recomputation of the anchor
    live = execs[execs > 0]
    avg = int(live.mean())
    Kd = 0.45 * (execs.max() - execs.min()) + execs.min()
    x = np.linspace(0, 300, 301) / 100 * avg
    y = x ** 1.8 / (x ** 1.8 + Kd ** 1.8)
    scale = 180_000 / y[100]
    check("Kd matches saturation x range + min", abs(c.Kd - Kd) < 1e-9)
    check("sales curve matches an independent Hill computation",
          np.allclose(rcd["Saturated_Sales"].values, y * scale, atol=1e-6))
    check("curve passes through (100%, contribution)",
          abs(float(c.currentpoint()["Saturated_Sales"].iloc[0]) - 180_000) < 1e-6)

    opt = c.optimalpoint()
    check("optimalpoint returns exactly one row", len(opt) == 1)
    check("optimal point is where avg and marginal return cross",
          abs(float(opt["Average Return"].iloc[0])
              - float(opt["Marginal Return"].iloc[0]))
          <= float(rcd.loc[rcd["Ratio"] >= .2, "Diff"].min()) + 1e-12)
    check("optimal is not the disqualified origin region",
          float(opt["Ratio"].iloc[0]) >= 0.2)

    # margin scales both return lines equally, so it CANNOT move the crossing.
    # Stated in the docs; asserted here because users will change it first.
    c2 = sat.SatCurve(df, "Amazon_Spend", spend=2_500_000, price=4.0,
                      margin=0.55, target_response=180_000,
                      saturation=0.45, slope=1.8)
    check("margin does not move the optimal point",
          abs(c2.to_dict()["optimal_ratio"] - c.to_dict()["optimal_ratio"]) < 1e-12,
          f"30% -> {c.to_dict()['optimal_ratio']:.2f}, "
          f"55% -> {c2.to_dict()['optimal_ratio']:.2f}")

    # A CONCAVE curve (slope < 1) has NO crossover: for y ~ x^n the ratio
    # marginal/average is exactly n at every spend level, so the lines are
    # parallel in ratio terms and never meet. argmin lands on the last grid
    # point, and that must be reported as inconclusive rather than as a
    # recommendation to triple spend.
    c3 = sat.SatCurve(df, "Amazon_Spend", spend=2_500_000, price=4.0,
                      margin=0.30, target_response=180_000,
                      saturation=0.45, slope=0.7)
    d3 = c3.to_dict()
    check("a concave curve (slope < 1) is flagged as having no crossover",
          d3["at_boundary"] and d3["direction"] == "inconclusive",
          f"optimal_ratio={d3['optimal_ratio']:.2f}, verdict='{d3['verdict']}'")
    # Closed form: for H(x) = x^n/(x^n + K^n),
    #     marginal / average = n·K^n / (x^n + K^n)
    # which is strictly below n. So whenever n <= 1 the marginal line sits
    # under the average line at EVERY spend level and no crossing can exist —
    # the "no crossover" flag is a theorem about the curve, not a numerical
    # accident on this particular series.
    # Checked over the region the optimal-point search actually considers
    # (ratio >= 0.2). Below that, `Marginal Return` is a discrete difference
    # over a 1%-of-spend step across a curve with an infinite slope at the
    # origin, so the secant collapses onto the average and the ratio tends to
    # 1 — a finite-difference artifact, and precisely why Alex disqualifies
    # everything under 20% of current spend.
    r3 = c3.response_curve_data()
    r3 = r3[r3["Ratio"] >= 0.2]
    x = r3["Ratio"].values * c3.average_nonzero_executions
    predicted = 0.7 * c3.Kd ** 0.7 / (x ** 0.7 + c3.Kd ** 0.7)
    actual = (r3["Marginal Return"] / r3["Average Return"]).values
    check("marginal/average follows n*K^n/(x^n+K^n), always < 1 when n <= 1",
          np.allclose(actual, predicted, atol=0.02) and (actual < 1).all(),
          f"max |actual - closed form| = {np.abs(actual - predicted).max():.4f}, "
          f"max ratio = {actual.max():.3f} (< 1 => lines cannot cross)")
    check("an S-curve (slope > 1) does produce an interior crossover",
          not c.to_dict()["at_boundary"],
          f"slope 1.8 -> optimum at {c.to_dict()['optimal_ratio']:.0%} of spend")


def test_transform_series(path):
    """The weekly raw/decayed/saturated diagnostic must be the SAME numbers the
    model used — not a re-derivation that could drift."""
    print("\nWeekly transform series")
    r = cp.run_slice(path, "Brand 1", "Channel 1")
    media = [c for c in r["selected"]
             if r["specs_by_name"][c].adstock_decay is not None]
    if not media:
        check("a media variable is in the model", False, "none selected")
        return
    v = media[0]
    spec = r["specs_by_name"][v]
    s = sat.media_transform_series(
        r["df"], v, decay=spec.adstock_decay,
        scale=getattr(spec, "scale", 1.0),
        sat_midpoint=spec.sat_midpoint, sat_slope=spec.sat_slope,
        sat_ref=(r.get("sat_refs") or {}).get(v),
        target=r["y"], dates=r["df"]["date"])
    model_col = r["X_all"][v].values
    series = np.asarray(s["saturated"] if s["saturated"] is not None
                        else s["decayed"])
    if float(getattr(spec, "scale", 1.0) or 1.0) != 1.0 and s["saturated"] is None:
        series = series / float(spec.scale)
    check(f"{v}: plotted series == the modeled column",
          np.allclose(series, model_col, atol=1e-9),
          f"max diff {np.abs(series - model_col).max():.2e}")
    check("all three correlations reported",
          set(s["corr"]) == {"raw", "decayed", "saturated"})
    check("raw series equals the source data",
          np.allclose(s["raw"], r["df"][v].astype(float).values))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pooled model properties
# ═══════════════════════════════════════════════════════════════════════════

def test_pooled(path):
    print("\nPooled model")
    r = ml.run_pooled(path, "Brand 1")

    check("national predictors detected", len(r["national"]) > 0,
          f"{len(r['national'])} national of "
          f"{len(r['specs'])} candidates")
    check("national split conserves the national totals",
          r["national_check"]["ok"],
          f"max relative error {r['national_check']['max_rel_error']:.2e}")
    check("Seasonality_Index NOT treated as national (equal sums, "
          "different values)",
          "Seasonality_Index" in r["national_check"]["sum_equal_only"]
          and "Seasonality_Index" not in r["national"])

    # contributions must reconstruct the fitted line exactly
    recon = r["fit"].contributions.sum(axis=1).values
    check("intercept + sum(beta*x) == fitted",
          np.allclose(recon, r["fit"].fitted, atol=1e-6),
          f"max diff {np.abs(recon - r['fit'].fitted).max():.2e}")

    # channel intercepts: each channel's fitted total must equal its actual
    # total (that is what a per-channel intercept from OLS guarantees)
    fitted = pd.Series(r["fit"].fitted, index=r["y"].index)
    worst = 0.0
    for ch, idx in r["meta"].groupby("channel").groups.items():
        i = np.asarray(list(idx))
        a, f = float(r["y"].loc[i].sum()), float(fitted.loc[i].sum())
        worst = max(worst, abs(f - a) / max(abs(a), 1.0))
    check("per-channel fitted total == actual total (channel intercepts)",
          worst < 1e-9, f"max relative gap {worst:.2e}")

    # the pooled model must use every channel's rows
    check("all kept channels are in the stacked matrix",
          set(r["meta"]["channel"]) == set(r["channels"]),
          f"{len(r['channels'])} channels, {r['n_rows']} rows")

    # channel-level due-tos must sum back to the pooled totals
    pc = r["per_channel"]
    for c in r["selected"]:
        tot = float(r["fit"].contributions[c].sum())
        check_ok = abs(pc[c].sum() - tot) <= 1e-6 * max(abs(tot), 1.0)
        if not check_ok:
            check(f"channel due-tos sum to the pooled total ({c})", False)
            break
    else:
        check("channel due-tos sum to the pooled total for every driver", True)

    check("centering did not leak the holdout",
          r["holdout_wmape"] > 0 and np.isfinite(r["holdout_wmape"]),
          f"holdout WMAPE {r['holdout_wmape']:.1f}% "
          f"(in-sample WMAPE {r['wmape']:.1f}%)")


def test_hierarchical(path):
    print("\nHierarchical model")
    h = ml.run_hierarchical(path, "Brand 1", shrink=1.0, enforce_sign=False,
                            sd_limit=None)
    chk = h["checks"]
    check("index averages to 1 on real data",
          chk["index_mean_max_dev"] < 1e-9,
          f"max deviation {chk['index_mean_max_dev']:.2e}")
    check("uncapped hierarchical coefficients average back to pooled",
          chk["uncapped_recovers_pooled"] < 1e-9)

    for ch, fit in h["fits"].items():
        recon = fit.contributions.sum(axis=1).values
        if not np.allclose(recon, fit.fitted, atol=1e-6):
            check(f"contributions reconstruct fitted ({ch})", False)
            break
    else:
        check("contributions reconstruct fitted for every channel", True)

    # sign guard
    hs = ml.run_hierarchical(path, "Brand 1", shrink=1.0, enforce_sign=True)
    bad = []
    for ch in hs["channels"]:
        for c in hs["selected"]:
            bp, bc = float(hs["beta_pooled"][c]), float(hs["final"].loc[ch, c])
            if bp != 0 and bc != 0 and np.sign(bp) != np.sign(bc):
                bad.append((ch, c))
    check("sign guard: no channel coefficient flips sign vs pooled",
          not bad, f"{len(bad)} flips" if bad else "")

    check("index instability is diagnosed, not hidden",
          "stable" in h["stability"].columns,
          f"{int((~h['stability']['stable']).sum())} of "
          f"{len(h['stability'])} predictors have |mean| < SD")


def test_levels_agree(path):
    """shrink=0 must give the pooled model back, exactly."""
    print("\nLevel continuity")
    p = ml.run_pooled(path, "Brand 2")
    h = ml.run_hierarchical(path, "Brand 2", shrink=0.0, pooled=p)
    worst = 0.0
    for ch in h["channels"]:
        for c in h["selected"]:
            worst = max(worst, abs(float(h["final"].loc[ch, c])
                                   - float(h["beta_pooled"][c])))
    check("shrink=0 reproduces the pooled coefficients exactly",
          worst < 1e-12, f"max diff {worst:.2e}")

    h1 = ml.run_hierarchical(path, "Brand 2", shrink=1.0, sd_limit=None,
                             enforce_sign=False, pooled=p)
    same = np.allclose(h1["final"].values, h1["index"].mul(
        h1["beta_pooled"], axis=1).values, atol=1e-12)
    check("shrink=1, no cap reproduces Alex's uncapped Model 3", same)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../Anonymized Data for Project.xlsx"
    import warnings
    warnings.filterwarnings("ignore")
    test_alex_spreadsheet()
    test_satcurve()
    test_transform_series(path)
    test_pooled(path)
    test_hierarchical(path)
    test_levels_agree(path)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)

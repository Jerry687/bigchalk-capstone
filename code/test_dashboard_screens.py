"""
End-to-end exercise of the three screens added on 2026-08-11 (High Level,
Saturation, Multi-Level).

Dash callbacks are ordinary functions, so they can be called directly with the
same arguments the browser would supply. That is what this does: build the
results cache once, then drive every new callback and assert the output is a
real component tree rather than an error banner — the failure mode a wiring
check alone cannot catch, because a callback with correct ids can still throw
the moment it touches data.

    cd code && python test_dashboard_screens.py
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

import dashboard as D                                           # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def _text(node, out=None):
    """Flatten a component tree to text so error banners can be spotted."""
    out = [] if out is None else out
    if node is None:
        return out
    if isinstance(node, str):
        out.append(node)
        return out
    if isinstance(node, (list, tuple)):
        for x in node:
            _text(x, out)
        return out
    ch = getattr(node, "children", None)
    if ch is not None:
        _text(ch, out)
    for attr in ("data", "columns"):
        v = getattr(node, attr, None)
        if isinstance(v, list):
            out.append(f"<{attr}:{len(v)}>")
    return out


def _has_error(node):
    t = " ".join(_text(node)).lower()
    return any(k in t for k in ("traceback", "could not", "error:",
                                "not in this product"))


def main(path):
    dp = {"path": path, "sheets": D.detect_product_sheets(path)}
    target, window = "Volume Sales", 104
    brand = dp["sheets"][0]

    print("\nBuilding the results cache")
    # Build under the SAME tuning signature the screens will look up with.
    # The cache is keyed by (datafile, target, window, tuning) precisely so a
    # result built under one set of selection hyperparameters is never served
    # for another — so a test that builds with tuning=None and reads with
    # tuning_of("balanced", ...) finds nothing, which is the cache working.
    tuning = D.tuning_of("balanced", None)
    store = D.build_all_results(path, target, window, tuning)
    check("cache built", len(store["models"]) > 0,
          f"{len(store['models'])} models, {len(store['skipped'])} skipped")
    channel = next(c for (b, c) in store["models"] if b == brand)

    # ── High Level ───────────────────────────────────────────────────────
    print("\nHigh Level (Total Brand / Total Channel)")
    for mode in ("brand", "channel", "all"):
        summary, opts, val, status = D._total_summary(
            mode, "total", dp, target, window, "balanced", None, None)
        check(f"{mode}: summary renders",
              summary is not None and not _has_error(summary)
              and len(opts) > 0, f"{len(opts)} groups")
        detail = D._total_detail(val, mode, dp, target, window,
                                 "balanced", None)
        check(f"{mode}: detail renders for '{val}'",
              detail is not None and not _has_error(detail))

    # the numbers behind the screen, checked directly
    import rollup as ru
    rolled = ru.rollup(list(store["models"].values()), "brand")
    worst = max(g["stats"]["decomposition_error"]
                for g in rolled["groups"].values())
    check("rolled-up contributions still add to the rolled-up fitted line",
          worst < 1e-9, f"max relative error {worst:.1e}")
    # totals must equal the sum of their parts
    tot = sum(float(np.sum(p["actual"])) for p in store["models"].values())
    rolled_all = ru.rollup(list(store["models"].values()), "all")
    got = rolled_all["groups"]["Total"]["stats"]["volume"]
    check("grand total volume == sum of every slice",
          abs(got - tot) / tot < 1e-12, f"{got:,.0f}")

    # ── Saturation ───────────────────────────────────────────────────────
    print("\nSaturation")
    opts, var, note = D._sat_vars("sat", brand, channel, dp, target, window,
                                  "balanced", None, None)
    if not opts:
        # find a slice that does have media so the screen can be exercised
        for (b, c), p in store["models"].items():
            m = (p.get("contrib_wk") or {}).get("media") or []
            if m:
                brand, channel = b, c
                opts, var, note = D._sat_vars("sat", brand, channel, dp,
                                              target, window, "balanced",
                                              None, None)
                break
    check("media variables offered", bool(opts),
          f"{brand} x {channel}: {len(opts)} media")

    dec, mid, slp = D._sat_defaults(var, 0, brand, channel, dp, target,
                                    window, "balanced", None)
    check("sliders default to the model's own settings",
          all(isinstance(v, float) for v in (dec, mid, slp)),
          f"decay={dec} midpoint={mid} slope={slp}")

    weekly, corr, status = D._sat_weekly(var, dec, mid, slp, brand, channel,
                                         dp, target, window, "balanced", None)
    check("weekly raw/decayed/saturated panel renders",
          weekly is not None and not _has_error(weekly), status)
    check("correlations panel renders",
          corr is not None and not _has_error(corr))

    # the override must actually change the picture
    w2, c2, s2 = D._sat_weekly(var, min(dec + 0.3, 0.9), mid, slp, brand,
                               channel, dp, target, window, "balanced", None)
    check("changing decay changes the output", s2 != status,
          f"'{status}' -> '{s2}'")
    check("an overridden setting is labelled as such", "overridden" in s2)

    spend, price, inote = D._sat_inputs(var, brand, channel, dp, target,
                                        window, "balanced", None)
    ok_inputs = (spend is not None and price is not None and spend > 0)
    check("spend and price auto-derive", ok_inputs,
          f"spend={spend:,.0f} price={price}" if ok_inputs
          else f"spend={spend} price={price}")

    roi = D._sat_roi(var, mid, slp, spend, price, 30, brand, channel, dp,
                     target, window, "balanced", None)
    check("ROI curve renders", roi is not None and not _has_error(roi),
          " ".join(_text(roi))[:90])

    # margin must not move the optimum (it scales both return lines)
    import saturation_curves as sc
    import capstone_pipeline as cp
    r = cp.run_slice(path, brand, channel, config=cp.ModelConfig(
        target=target, model_weeks=window,
        window_end=D.dataset_anchor(path),
        variable_config=cp.resolve_config_path(path, brand, channel)))
    a = sc.build_sat_curve(r, var, margin=0.30)
    b = sc.build_sat_curve(r, var, margin=0.60)
    if "error" not in a and "error" not in b:
        check("margin does not move the optimal point on real data",
              abs(a["optimal_ratio"] - b["optimal_ratio"]) < 1e-12,
              f"optimal at {a['optimal_ratio']:.0%} either way")
        check("optimalpoint() is reported",
              "optimal_spend" in a and "optimal_marginal_return" in a)

    # ── Multi-Level ──────────────────────────────────────────────────────
    print("\nMulti-Level")
    out = D._multi_run(1, brand, dp, target, window, 1.0, 1.0,
                       ["fit", "sign", "fe"])
    check("all three levels run and render",
          out is not None and not _has_error(out),
          " ".join(_text(out))[:100])

    out2 = D._multi_run(1, brand, dp, target, window, 0.0, 1.0, ["sign", "fe"])
    check("manual lambda (no holdout fit) also renders",
          out2 is not None and not _has_error(out2))

    note = D._multi_note(brand, "multi")
    check("product note reflects the top-bar selection", brand in str(note))

    # ── tab plumbing ─────────────────────────────────────────────────────
    print("\nNavigation")
    n = len(D.SCREENS)
    for key, label in D.SCREENS:
        res = D._render_tabs(key)
        styles, classes = res[:n], res[n:]
        shown = [D.SCREENS[i][0] for i, s in enumerate(styles) if s == {}]
        if shown != [key]:
            check(f"'{label}' shows exactly its own screen", False, str(shown))
            break
    else:
        check(f"all {n} screens show exactly one panel each", True,
              ", ".join(l for _, l in D.SCREENS))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "../Anonymized Data for Project.xlsx"
    main(p)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)

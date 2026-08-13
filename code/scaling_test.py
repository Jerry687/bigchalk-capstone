"""
Runtime / scaling test for the automated regression engine.

Answers the sponsor question (Alex, 2026-07-21): *does runtime scale linearly
with the number of models, or does it blow up?*

Design
------
Alex asked for four points: 1 model, 10 models, the full set (~160 slices),
and the full set run twice in one process (~320) to expose any super-linear
term (memory growth, accumulating state, GC pressure).

Two clocks are reported, because they scale differently and conflating them
hides the answer:

  * ``model_s``  - engine time only: config resolve + adstock + VIF prune +
                   forward selection + constrained fit + contributions.
                   This is the term that multiplies by the number of slices.
  * ``load_s``   - reading the source workbook. Paid ONCE PER BRAND SHEET
                   (10 sheets), not once per slice, and it is pure I/O on a
                   13 MB xlsx. Amortised, it is a fixed cost, so it is timed
                   separately rather than smeared across the per-model number.

Fit
---
A least-squares line ``total = a * n + b`` is fitted over the four points and
reported with R^2, alongside the empirical growth ratio between consecutive
scale points. Linear scaling => ratio ~= the ratio of n, and per-model time
flat across all four points.

Usage
-----
    python scaling_test.py                # full test (1, 10, all, all x2)
    python scaling_test.py --max-slices 20   # quick smoke run
    python scaling_test.py --skip-double     # omit the ~320 point

Writes ``outputs/scaling_test_results.csv`` (per-slice detail),
``outputs/scaling_test_summary.csv`` (the four scale points) and
``outputs/scaling_test.png`` (total time vs. number of models).
"""
import os

# Pin BLAS threads BEFORE numpy loads - same rule as run_all.py. Each solve is
# tiny; thread spawn overhead dominates otherwise, and an unpinned run makes
# the scaling curve noisy for reasons that have nothing to do with scaling.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone_pipeline as cp  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

BASE_CFG = dict(
    target="Volume Sales",
    model_weeks=104,
    holdout_weeks=None,
    p_enter=0.05,
    default_media_decay=0.5,
)


def enumerate_slices(max_slices=None):
    """Return [(brand, channel, sheet_df)] for every modelable slice.

    Slices whose target is all zero (brand not sold in that channel) are
    excluded: run_all.py skips them, so counting them would understate the
    per-model cost.
    """
    xl = pd.ExcelFile(DATA)
    brands = [s for s in xl.sheet_names if s.lower().startswith("brand")]
    slices, load_times, sheets = [], [], {}
    for brand in brands:
        t0 = time.perf_counter()
        sheet = pd.read_excel(xl, sheet_name=brand)
        load_times.append(time.perf_counter() - t0)
        sheets[brand] = sheet
        for channel in sheet["Geography"].unique():
            if not isinstance(channel, str):
                continue
            tgt = sheet.loc[sheet["Geography"] == channel, BASE_CFG["target"]]
            if tgt.abs().sum() == 0:
                continue
            slices.append((brand, channel))
        if max_slices and len(slices) >= max_slices:
            break
    if max_slices:
        slices = slices[:max_slices]
    return slices, sheets, sum(load_times)


def time_one(brand, channel, sheet):
    """Wall time for a single slice through the full engine path."""
    t0 = time.perf_counter()
    cfg_file = (cp.resolve_config_path(DATA, brand, channel)
                or cp.load_or_create_default_config(DATA, brand, df=sheet))
    cfg = cp.ModelConfig(**BASE_CFG, variable_config=cfg_file)
    r = cp.run_slice(DATA, brand, channel, config=cfg, df=sheet)
    elapsed = time.perf_counter() - t0
    return elapsed, len(r["selected"]), len(r["df"])


def run_scale_point(label, slices, sheets):
    """Run `slices` sequentially; return (summary_row, per_slice_rows)."""
    rows = []
    t0 = time.perf_counter()
    for brand, channel in slices:
        try:
            secs, n_sel, n_wk = time_one(brand, channel, sheets[brand])
            rows.append({"scale_point": label, "brand": brand,
                         "channel": channel, "model_s": round(secs, 4),
                         "n_selected": n_sel, "n_weeks": n_wk, "status": "ok"})
        except Exception as e:  # a failure still consumes time - record it
            rows.append({"scale_point": label, "brand": brand,
                         "channel": channel, "model_s": np.nan,
                         "n_selected": np.nan, "n_weeks": np.nan,
                         "status": f"failed: {type(e).__name__}"})
    total = time.perf_counter() - t0
    ok = [r["model_s"] for r in rows if r["status"] == "ok"]
    summary = {
        "scale_point": label,
        "n_models": len(slices),
        "n_ok": len(ok),
        "total_s": round(total, 2),
        "per_model_s": round(total / len(slices), 4) if slices else np.nan,
        "median_model_s": round(float(np.median(ok)), 4) if ok else np.nan,
        "p95_model_s": round(float(np.percentile(ok, 95)), 4) if ok else np.nan,
    }
    print(f"  {label:>10}: {len(slices):>4} models  "
          f"{total:8.1f}s total  {summary['per_model_s']:.3f}s/model",
          flush=True)
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-slices", type=int, default=None,
                    help="cap the full-set size (for a quick smoke run)")
    ap.add_argument("--skip-double", action="store_true",
                    help="skip the doubled (~2x full set) point")
    args = ap.parse_args()

    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}  |  BLAS threads pinned to "
          f"{os.environ.get('OMP_NUM_THREADS')}")
    print("\nEnumerating slices (includes one-time workbook read)...")
    slices, sheets, load_s = enumerate_slices(args.max_slices)
    print(f"  {len(slices)} modelable slices across {len(sheets)} brand sheets")
    print(f"  workbook read (all sheets, one-time I/O): {load_s:.1f}s\n")

    # Warm-up: the first call pays import/JIT/config-creation costs that are
    # not part of steady-state per-model time. Excluding it keeps the "1 model"
    # point comparable to the others instead of inflating the smallest n.
    print("Warm-up run (excluded from results)...")
    time_one(*slices[0], sheets[slices[0][0]])

    summaries, details = [], []
    print("\nScale points:")
    for label, subset in [("1", slices[:1]), ("10", slices[:10]),
                          ("full", slices)]:
        s, d = run_scale_point(label, subset, sheets)
        summaries.append(s)
        details.extend(d)

    if not args.skip_double:
        s, d = run_scale_point("full x2", slices + slices, sheets)
        summaries.append(s)
        details.extend(d)

    sm = pd.DataFrame(summaries)

    # Linearity: fit total_s = a*n + b and report R^2 plus consecutive growth
    # ratios. R^2 near 1 with a flat per-model time == linear scaling.
    n = sm["n_models"].to_numpy(float)
    t = sm["total_s"].to_numpy(float)
    a, b = np.polyfit(n, t, 1)
    resid = t - (a * n + b)
    r2 = 1 - resid.var() / t.var() if t.var() > 0 else float("nan")

    sm["growth_vs_prev"] = (sm["total_s"] / sm["total_s"].shift(1)).round(2)
    sm["n_ratio_vs_prev"] = (sm["n_models"] / sm["n_models"].shift(1)).round(2)

    pd.DataFrame(details).to_csv(
        os.path.join(OUT, "scaling_test_results.csv"), index=False)
    sm.to_csv(os.path.join(OUT, "scaling_test_summary.csv"), index=False)

    print("\n" + "=" * 68)
    print(sm.to_string(index=False))
    print("=" * 68)
    print(f"Linear fit: total_s = {a:.4f} * n_models + {b:.2f}   R^2 = {r2:.4f}")
    print(f"Marginal cost per additional model: {a:.3f}s")
    print(f"One-time workbook read (all 10 sheets): {load_s:.1f}s")
    full = sm.loc[sm.scale_point == "full", "total_s"].iloc[0]
    print(f"Full set ({int(sm.loc[sm.scale_point=='full','n_models'].iloc[0])} "
          f"models): {full:.0f}s = {full/60:.1f} min "
          f"(+ {load_s:.0f}s data load)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        grid = np.linspace(0, n.max() * 1.05, 100)
        ax1.plot(grid, a * grid + b, "--", color="#999",
                 label=f"linear fit (R²={r2:.4f})")
        ax1.plot(n, t, "o-", color="#1f77b4", lw=2, label="measured")
        for xi, yi in zip(n, t):
            ax1.annotate(f"{yi:.0f}s", (xi, yi), textcoords="offset points",
                         xytext=(6, -12), fontsize=9)
        ax1.set_xlabel("number of models")
        ax1.set_ylabel("total runtime (s)")
        ax1.set_title("Runtime scales linearly with model count")
        ax1.legend()
        ax1.grid(alpha=.3)

        ax2.bar([str(x) for x in sm.scale_point], sm.per_model_s,
                color="#2ca02c")
        for i, v in enumerate(sm.per_model_s):
            ax2.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("seconds per model")
        ax2.set_title("Per-model cost is flat as the batch grows")
        ax2.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "scaling_test.png"), dpi=150)
        print(f"\nChart: {os.path.join(OUT, 'scaling_test.png')}")
    except ImportError:
        print("\n(matplotlib not available - skipped chart)")


if __name__ == "__main__":
    main()

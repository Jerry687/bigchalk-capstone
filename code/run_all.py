"""
Phase 3: batch-run the regression engine over every Brand x Channel slice.

For each slice: full pipeline (config-driven selection + constrained fit +
holdout validation), per-slice coefficient/contribution exports, and one
cross-slice summary table for spotting slices that model poorly.

Outputs:
  ../outputs/all/<brand>_<channel>_coefficients.csv
  ../outputs/all/<brand>_<channel>_contrib_by_year.csv
  ../outputs/all_models_summary.csv

Note: ACV Weighted Distribution is force-included by DEFAULT — the generated
default config sets its ROLE to `force` (a volume model with no distribution
term cannot track shelf-presence change; validated on Brand 1 x Channel 1:
holdout MAPE 73% -> 32%). It is NOT hard-forced here: an analyst can override
it to `auto` per Product/Channel and the config role is authoritative for both
CLI and web app. Per-slice tuning lives in the two-tier configs under configs/
(Product default + optional Product x Channel override), resolved via
cp.resolve_config_path — the same rule the dashboard uses.
"""
import os
# Pin BLAS threads BEFORE numpy loads: repeated small least-squares solves
# thrash badly with many threads (each solve is tiny; spawn overhead dominates).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import time
import traceback

import numpy as np
import pandas as pd

import capstone_pipeline as cp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
# Write to fast local scratch during the run (network/synced folders can hang
# on many small writes), then copy into the project at the end.
SCRATCH = os.environ.get("BATCH_SCRATCH")
OUT = SCRATCH if SCRATCH else os.path.join(ROOT, "outputs")
OUT_ALL = os.path.join(OUT, "all")
os.makedirs(OUT_ALL, exist_ok=True)

BASE_CFG = dict(
    target="Volume Sales",
    model_weeks=104,
    holdout_weeks=None,          # auto: always-reserved validation tail (<=13w)
    p_enter=0.05,
    default_media_decay=0.5,
    # Per-family selection caps (Alex 2026-07-27: "only two to three trade,
    # but all media"). Empty = no caps = the historical behaviour; set e.g.
    # {"Trade": 3, "Competitive": 2, "Macro": 2} to try a tighter model.
    max_per_family={},
    # NOTE: no force_include here. ACV force-include lives in the config file's
    # ROLE (the generated default forces it), so it's authoritative and the
    # analyst can override it to `auto`. Passing force_include on top would let
    # the CLI disagree with the web app for such an override.
)


def slug(s: str) -> str:
    return s.replace(" ", "").lower()


def main():
    import sys
    xl = pd.ExcelFile(DATA)
    brands = [s for s in xl.sheet_names if s.lower().startswith("brand")]
    # optional chunking: `python run_all.py 1 3` runs Brand 1..Brand 3 only
    # (lets long batches run in resumable chunks); summary parts are merged
    # by summarize().
    part = ""
    if len(sys.argv) == 3:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
        brands = [b for b in brands if lo <= int(b.split()[-1]) <= hi]
        part = f"_part{lo}_{hi}"
    rows, failures, skipped = [], [], []
    t0 = time.time()

    # ONE fixed calendar window for the whole batch: the latest week in the
    # DATASET, not in each slice (Alex 2026-07-27). Every model below covers
    # the identical date range, so slices are comparable and a delisted slice
    # cannot smuggle 2023 history into the exports.
    anchor = cp.dataset_anchor_week(DATA)
    win_start, win_end = cp.resolve_window(anchor, BASE_CFG["model_weeks"])
    print(f"Modeling window: {win_start:%Y-%m-%d} .. {win_end:%Y-%m-%d} "
          f"({BASE_CFG['model_weeks']} wks, anchored to the dataset's latest "
          f"week)", flush=True)

    for brand in brands:
        sheet = pd.read_excel(xl, sheet_name=brand)
        channels = [c for c in sheet["Geography"].unique() if isinstance(c, str)]
        for channel in channels:
            tag = f"{brand} x {channel}"
            # brands not sold in a channel have an all-zero target: skip
            if sheet.loc[sheet["Geography"] == channel, BASE_CFG["target"]].abs().sum() == 0:
                failures.append({"brand": brand, "channel": channel,
                                 "error": "target all zero - not sold in channel (skipped)"})
                print(f"{tag}: SKIPPED (target all zero)", flush=True)
                continue
            try:
                # Two-tier config: Product × Channel override wins, else the
                # Product default (created if missing). Always a config file, so
                # roles (incl. ACV force) are authoritative and identical to the
                # web app — no separate force_include that could diverge.
                cfg_file = (cp.resolve_config_path(DATA, brand, channel)
                            or cp.load_or_create_default_config(
                                DATA, brand, df=sheet))
                cfg = cp.ModelConfig(**BASE_CFG, variable_config=cfg_file,
                                     window_end=anchor)
                r = cp.run_slice(DATA, brand, channel, config=cfg, df=sheet)
                fit, sel = r["fit"], r["selected"]

                contrib_means = {c: r["avg_contrib"].loc[c, "avg_weekly_contribution"]
                                 for c in sel}
                top_pos = max(contrib_means, key=contrib_means.get)
                top_neg = min(contrib_means, key=contrib_means.get)

                rows.append({
                    "brand": brand, "channel": channel,
                    "window_start": f"{r['window_start']:%Y-%m-%d}",
                    "window_end": f"{r['window_end']:%Y-%m-%d}",
                    "n_weeks_reported": len(r["df"]),
                    "n_weeks_selling": r["n_weeks_selling"],
                    "n_weeks_holdout": len(r["yte"]),
                    "n_selected": len(sel), "n_forced": len(r["forced"]),
                    "R2": round(fit.r2, 4), "adj_R2": round(fit.adj_r2, 4),
                    "MAPE_in_pct": round(fit.mape, 2),
                    "MAPE_holdout_pct": round(r["holdout_mape"], 2),
                    "durbin_watson": round(fit.meta["durbin_watson"], 3),
                    "n_sign_conflicts": len(r["sign_conflicts"]),
                    "top_positive_driver": top_pos,
                    "top_negative_driver": top_neg,
                })

                base = f"{OUT_ALL}/{slug(brand)}_{slug(channel)}"
                coef = pd.DataFrame({
                    "coefficient": fit.coef,
                    "family": ["(intercept)"] + [r["specs_by_name"][c].family for c in sel],
                    "sign": ["(intercept)"] + [r["specs_by_name"][c].sign for c in sel],
                    "forced": [""] + ["yes" if c in r["forced"] else "" for c in sel],
                    "t_stat_OLS": [np.nan] + [fit.tstats.get(c, np.nan) for c in sel],
                    "avg_weekly_contribution":
                        [r["avg_contrib"].loc["Intercept", "avg_weekly_contribution"]]
                        + [r["avg_contrib"].loc[c, "avg_weekly_contribution"] for c in sel],
                })
                coef.to_csv(f"{base}_coefficients.csv")
                r["contrib_by_year"].to_csv(f"{base}_contrib_by_year.csv")
                print(f"[{time.time()-t0:6.0f}s] {tag}: R2={fit.r2:.2f} "
                      f"holdout MAPE={r['holdout_mape']:.0f}%", flush=True)
            # Not enough data INSIDE the fixed window -> deliberately NOT
            # modeled. This is an expected outcome, not a crash: it is how a
            # delisted / barely-distributed slice stops producing a phantom
            # model. Reported separately so nothing disappears silently.
            except cp.InsufficientWindowData as e:
                skipped.append({"brand": brand, "channel": channel,
                                "weeks_in_window": e.weeks,
                                "selling_weeks_in_window": e.nonzero,
                                "reason": str(e)})
                print(f"[{time.time()-t0:6.0f}s] {tag}: SKIPPED - {e}",
                      flush=True)
            except Exception as e:
                failures.append({"brand": brand, "channel": channel, "error": str(e)})
                print(f"[{time.time()-t0:6.0f}s] {tag}: FAILED - {e}", flush=True)
                traceback.print_exc()

    summary = pd.DataFrame(rows).sort_values("MAPE_holdout_pct")
    summary.to_csv(f"{OUT}/all_models_summary{part}.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(f"{OUT}/all_models_failures{part}.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(f"{OUT}/all_models_skipped{part}.csv",
                                     index=False)

    print(f"\nDone: {len(rows)} models, {len(skipped)} skipped "
          f"(insufficient data in window), {len(failures)} failures "
          f"in {time.time()-t0:.0f}s")
    print(f"median R2={summary.R2.median():.2f}  "
          f"median holdout MAPE={summary.MAPE_holdout_pct.median():.1f}%")


if __name__ == "__main__":
    main()

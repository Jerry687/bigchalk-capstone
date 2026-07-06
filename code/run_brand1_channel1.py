"""
Runner for Brand 1 x Channel 1 (updated per Alex's sponsor review, 2026-07-06).
Generates EDA charts, the constrained model, diagnostics, and result exports.
Everything is driven by ModelConfig + variable_config.csv — no hard-coded
target names or variable lists. Outputs land in ../outputs/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import capstone_pipeline as cp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
OUT = os.path.join(ROOT, "outputs")
VARCFG = os.path.join(ROOT, "variable_config.csv")
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
BRAND, CHANNEL = "Brand 1", "Channel 1"

# --- run-level configuration (everything Alex asked to be adjustable) ---
CFG = cp.ModelConfig(
    target="Volume Sales",   # any raw column can be the dependent variable
    model_weeks=104,         # set-year window (52 / 104 / 156)
    holdout_weeks=None,      # None -> weeks beyond the window (113-104 = 9)
    p_enter=0.05,
    default_media_decay=0.5,
    force_include=[],        # client-mandated variables go here
    exclude=[],
    variable_config=VARCFG if os.path.exists(VARCFG) else None,
)


def main():
    df = cp.load_slice(DATA, BRAND, CHANNEL)
    y = df[CFG.target]
    tname = CFG.target

    # Write the editable variable-config template on first run; afterwards
    # the CSV (families / signs / bounds / decays / roles) is source of truth.
    if not os.path.exists(VARCFG):
        cp.generate_variable_config(df, VARCFG, CFG.default_media_decay)
        CFG.variable_config = VARCFG
        print(f"Wrote variable-config template -> {VARCFG}")

    # ---------------- EDA ----------------
    # 1. Target over time
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], y, color="#2b6cb0", lw=2)
    ax.set_title(f"{tname} over time — {BRAND} x {CHANNEL}")
    ax.set_ylabel(tname); ax.set_xlabel("Week")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_01_target_timeseries.png", dpi=130); plt.close(fig)

    # 2. Seasonality: average by week-of-year (follows whatever the target is)
    if "Week_Num" in df:
        wk = df.groupby("Week_Num")[tname].mean()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(wk.index, wk.values, color="#dd6b20", lw=2)
        ax.set_title(f"Average {tname} by week of year (seasonality)")
        ax.set_xlabel("Week number (1–52)"); ax.set_ylabel(f"Mean {tname}")
        fig.tight_layout(); fig.savefig(f"{OUT}/eda_02_seasonality.png", dpi=130); plt.close(fig)

    # 3. Target distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(y, kde=True, color="#38a169", ax=ax)
    ax.set_title(f"Distribution of {tname}")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_03_target_hist.png", dpi=130); plt.close(fig)

    # 4. Correlation of candidate (non-leakage) predictors with target
    specs = cp.build_feature_specs(df, config=CFG.variable_config)
    Xcand = cp.assemble_matrix(df, specs)
    corr_t = Xcand.apply(lambda c: c.corr(y)).sort_values(key=abs, ascending=False)
    top = corr_t.head(18)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="vlag")
    ax.set_title(f"Top candidate predictors by |corr| with {tname}")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_04_predictor_corr.png", dpi=130); plt.close(fig)

    # ---------------- MODEL ----------------
    r = cp.run_slice(DATA, BRAND, CHANNEL, config=CFG)
    fit = r["fit"]; selected = r["selected"]; dfm = r["df"]; ym = r["y"]

    # adstock sanity: decayed totals ~= raw totals (normalized adstock)
    print("\nAdstock totals check (adstocked/raw, ~1.0 expected):")
    for cname, chk in sorted(r["adstock_checks"].items()):
        if chk["raw_total"] > 0:
            print(f"  {cname}: {chk['ratio']:.3f}")

    # 5. Actual vs fitted (training window) + holdout forecast
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dfm["date"], ym, label="Actual", color="#2d3748", lw=2)
    ax.plot(dfm["date"].iloc[:r["split"]], fit.fitted, label="Fitted",
            color="#e53e3e", lw=2, ls="--")
    if len(r["pred_te"]):
        ax.plot(dfm["date"].iloc[r["split"]:], r["pred_te"], label="Holdout forecast",
                color="#d69e2e", lw=2, ls="--")
        ax.axvline(dfm["date"].iloc[r["split"]], color="grey", ls=":")
    ax.set_title(f"Actual vs Fitted — {BRAND} x {CHANNEL}  "
                 f"(R²={fit.r2:.2f}, MAPE={fit.mape:.1f}%)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/model_01_actual_vs_fitted.png", dpi=130); plt.close(fig)

    # 6. Residual diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(fit.fitted, fit.resid, alpha=0.6, color="#3182ce")
    axes[0].axhline(0, color="red", ls="--"); axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Fitted"); axes[0].set_ylabel("Residual")
    sns.histplot(fit.resid, kde=True, ax=axes[1], color="#805ad5")
    axes[1].set_title("Residual distribution")
    fig.tight_layout(); fig.savefig(f"{OUT}/model_02_residuals.png", dpi=130); plt.close(fig)

    # 7. Average weekly SIGNED contribution per driver. Media averaged only
    # over weeks with execution (never over the zeros of a mid-window flight).
    avg = r["avg_contrib"]["avg_weekly_contribution"].drop("Intercept")
    avg = avg.sort_values()
    colors = ["#e53e3e" if v < 0 else "#4c51bf" for v in avg.values]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(avg.index, avg.values, color=colors)
    ax.axvline(0, color="#2d3748", lw=1)
    ax.set_title(f"Avg weekly due-to per driver (signed, {tname})\n"
                 "media averaged over execution weeks only; intercept excluded")
    ax.set_xlabel(f"Average weekly contribution ({tname} units)")
    fig.tight_layout(); fig.savefig(f"{OUT}/model_03_contributions.png", dpi=130); plt.close(fig)

    # 8. Contributions by model year (signed sums + YoY) — the CPG view:
    # how much better/worse is this year vs last, and due to what.
    cby = r["contrib_by_year"]
    year_cols = [c for c in cby.columns if not c.startswith("YoY")]
    plot_tbl = cby[year_cols].drop("Intercept")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tbl.plot(kind="barh", ax=ax, color=["#a0aec0", "#4c51bf"][-len(year_cols):])
    ax.axvline(0, color="#2d3748", lw=1)
    ax.set_title(f"Due-to totals by model year (signed, {tname})")
    ax.set_xlabel(f"Total contribution ({tname} units)")
    fig.tight_layout(); fig.savefig(f"{OUT}/model_04_contrib_by_year.png", dpi=130); plt.close(fig)

    # ---------------- EXPORTS ----------------
    coef_tbl = pd.DataFrame({
        "coefficient": fit.coef,
        "expected_sign": ["(intercept)"] + [r["specs_by_name"][c].sign for c in selected],
        "family": ["(intercept)"] + [r["specs_by_name"][c].family for c in selected],
        "forced": [""] + ["yes" if c in r["forced"] else "" for c in selected],
        "t_stat_OLS": [np.nan] + [fit.tstats.get(c, np.nan) for c in selected],
        "VIF": [np.nan] + [fit.vif.get(c, np.nan) for c in selected],
        "avg_weekly_contribution":
            [r["avg_contrib"].loc["Intercept", "avg_weekly_contribution"]]
            + [r["avg_contrib"].loc[c, "avg_weekly_contribution"] for c in selected],
    })
    coef_tbl.to_csv(f"{OUT}/model_coefficients.csv")
    cby.to_csv(f"{OUT}/model_contributions_by_year.csv")

    summary = {
        "brand": BRAND, "channel": CHANNEL, "target": tname,
        "model_weeks": r["split"], "holdout_weeks": len(r["yte"]),
        "n_predictors_selected": len(selected),
        "n_forced": len(r["forced"]),
        "R2": round(fit.r2, 4), "adj_R2": round(fit.adj_r2, 4),
        "in_sample_MAPE_pct": round(fit.mape, 2),
        "holdout_MAPE_pct": round(r["holdout_mape"], 2),
        "durbin_watson": round(fit.meta["durbin_watson"], 3),
    }
    pd.Series(summary).to_csv(f"{OUT}/model_summary.csv")

    # contribution sanity check: intercept + due-tos reconstruct fitted exactly
    recon = fit.contributions.sum(axis=1).values
    max_err = float(np.max(np.abs(recon - fit.fitted)))

    print(f"\n=== {BRAND} x {CHANNEL} (target: {tname}) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  contribution reconstruction max error: {max_err:.2e}")
    neg = [c for c in selected if coef_tbl.loc[c, 'avg_weekly_contribution'] < 0]
    print(f"  drivers with negative due-tos: {neg if neg else 'NONE — double-check!'}")
    print("\nSelected predictors & coefficients:")
    print(coef_tbl.round(3).to_string())
    print("\nDue-tos by model year:")
    print(cby.round(0).to_string())
    if r["sign_conflicts"]:
        print("\nSIGN CONFLICTS (prior vs data) -- discuss with Big Chalk:")
        for c, prior, msg, t in r["sign_conflicts"]:
            print(f"  {c}: prior={prior}, {msg} (OLS t={t:.2f})")
    else:
        print("\nNo sign conflicts: all selected predictors agree with priors.")


if __name__ == "__main__":
    main()

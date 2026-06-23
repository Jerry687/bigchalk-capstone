"""
Phase 1 + 2 runner for Brand 1 x Channel 1.
Generates EDA charts, the constrained model, diagnostics, and result exports.
Outputs land in ../outputs/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import capstone_pipeline as cp

DATA = os.path.join(os.path.dirname(__file__), "..", "Anonymized Data for Project.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
BRAND, CHANNEL = "Brand 1", "Channel 1"


def main():
    df = cp.load_slice(DATA, BRAND, CHANNEL)
    y = df[cp.TARGET]

    # ---------------- EDA ----------------
    # 1. Target over time
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], y, color="#2b6cb0", lw=2)
    ax.set_title(f"Volume Sales over time — {BRAND} x {CHANNEL}")
    ax.set_ylabel("Volume Sales"); ax.set_xlabel("Week")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_01_target_timeseries.png", dpi=130); plt.close(fig)

    # 2. Seasonality: average by week-of-year
    if "Week_Num" in df:
        wk = df.groupby("Week_Num")[cp.TARGET].mean()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(wk.index, wk.values, color="#dd6b20", lw=2)
        ax.set_title("Average Volume Sales by week of year (seasonality)")
        ax.set_xlabel("Week number (1–52)"); ax.set_ylabel("Mean Volume Sales")
        fig.tight_layout(); fig.savefig(f"{OUT}/eda_02_seasonality.png", dpi=130); plt.close(fig)

    # 3. Target distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(y, kde=True, color="#38a169", ax=ax)
    ax.set_title("Distribution of Volume Sales")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_03_target_hist.png", dpi=130); plt.close(fig)

    # 4. Correlation heatmap of candidate (non-leakage) predictors with target
    specs = cp.build_feature_specs(df)
    Xcand = cp.assemble_matrix(df, specs)
    corr_t = Xcand.apply(lambda c: c.corr(y)).sort_values(key=abs, ascending=False)
    top = corr_t.head(18)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="vlag")
    ax.set_title("Top candidate predictors by |corr| with Volume Sales")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout(); fig.savefig(f"{OUT}/eda_04_predictor_corr.png", dpi=130); plt.close(fig)

    # ---------------- MODEL ----------------
    r = cp.run_slice(DATA, BRAND, CHANNEL)
    fit = r["fit"]; selected = r["selected"]

    # 5. Actual vs fitted
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], y, label="Actual", color="#2d3748", lw=2)
    ax.plot(df["date"], fit.fitted, label="Fitted", color="#e53e3e", lw=2, ls="--")
    ax.axvline(df["date"].iloc[r["split"]], color="grey", ls=":", label="Holdout start")
    ax.set_title(f"Actual vs Fitted — {BRAND} x {CHANNEL}  (R²={fit.r2:.2f}, MAPE={fit.mape:.1f}%)")
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

    # 7. Demeaned contribution (due-to) decomposition.
    # Reads as deviation-from-average sales explained by each driver, so a single
    # large-magnitude level series cannot visually dominate the baseline.
    Xsel = r["X_all"][selected]
    demeaned = pd.DataFrame({c: fit.coef[c] * (Xsel[c].values - Xsel[c].mean())
                             for c in selected}, index=Xsel.index)
    avg_abs = demeaned.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(avg_abs.index, avg_abs.values, color="#4c51bf")
    ax.set_title("Driver importance — mean |deviation contribution| to Volume Sales")
    ax.set_xlabel("Mean absolute contribution to weekly deviation (volume units)")
    fig.tight_layout(); fig.savefig(f"{OUT}/model_03_contributions.png", dpi=130); plt.close(fig)

    # ---------------- EXPORTS ----------------
    coef_tbl = pd.DataFrame({
        "coefficient": fit.coef,
        "expected_sign": ["(intercept)"] + [r["specs_by_name"][c].sign for c in selected],
        "family": ["(intercept)"] + [r["specs_by_name"][c].family for c in selected],
        "t_stat_OLS": [np.nan] + [fit.tstats.get(c, np.nan) for c in selected],
        "VIF": [np.nan] + [fit.vif.get(c, np.nan) for c in selected],
        "avg_contribution": [fit.contributions["Base (const)"].mean()]
                            + [fit.contributions[c].mean() for c in selected],
    })
    coef_tbl.to_csv(f"{OUT}/model_coefficients.csv")

    summary = {
        "brand": BRAND, "channel": CHANNEL, "n_weeks": len(df),
        "n_predictors_selected": len(selected),
        "R2": round(fit.r2, 4), "adj_R2": round(fit.adj_r2, 4),
        "in_sample_MAPE_pct": round(fit.mape, 2),
        "holdout_MAPE_pct": round(r["holdout_mape"], 2),
        "durbin_watson": round(fit.meta["durbin_watson"], 3),
    }
    pd.Series(summary).to_csv(f"{OUT}/model_summary.csv")

    # contribution sanity check
    recon = fit.contributions.sum(axis=1).values
    max_err = float(np.max(np.abs(recon - fit.fitted)))


    print("=== Brand 1 x Channel 1 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  contribution reconstruction max error: {max_err:.2e}")
    print("\nSelected predictors & coefficients:")
    print(coef_tbl.round(3).to_string())
    if r["sign_conflicts"]:
        print("\nSIGN CONFLICTS (prior vs data) -- discuss with Big Chalk:")
        for c, prior, msg, t in r["sign_conflicts"]:
            print(f"  {c}: prior={prior}, {msg} (OLS t={t:.2f})")
    else:
        print("\nNo sign conflicts: all selected predictors agree with priors.")


if __name__ == "__main__":
    main()

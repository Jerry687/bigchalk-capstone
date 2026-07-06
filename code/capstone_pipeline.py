"""
Big Chalk Capstone - Automated Regression Engine (core pipeline)
================================================================

Reusable, modular building blocks for modeling weekly Volume Sales for a single
Brand x Channel slice, designed to scale to every Brand x Channel combination
in Phase 3 and to plug into the Phase 4 dashboard.

Design principles
-----------------
1. No target leakage.
2. Marketing-mix framing.
3. Two-stage estimation:
       (a) automatedelection
       (b) constrained FINAL fit
4. Adstock on media

Author: Capstone team (Feifan, Boqi, Jiahao) | June 2026
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 1. Data loading

def _parse_week(value) -> Optional[datetime]:
    """Extract a date from strings like 'WE 01/08/2023'."""
    m = re.search(r"(\d{2}/\d{2}/\d{4})", str(value))
    return datetime.strptime(m.group(1), "%m/%d/%Y") if m else None


def load_slice(path: str, brand_sheet: str, channel: str) -> pd.DataFrame:
    """Load one Brand x Channel slice, sorted by week, with a parsed `date`."""
    df = pd.read_excel(path, sheet_name=brand_sheet)
    df = df[df["Geography"] == channel].copy()
    df["date"] = df["Time"].apply(_parse_week)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# 2. Transforms

def adstock(x: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Geometric adstock: a_t = x_t + decay * a_{t-1}. Captures media carryover."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


# 3. Predictor specification (families + expected signs)

# Columns that are decompositions of the target -> NEVER use as predictors.
LEAKAGE_PREFIXES = ("Volume Sales", "Dollar Sales")
TARGET = "Volume Sales"


@dataclass
class FeatureSpec:
    """One candidate predictor: source column, family, expected sign, transform."""
    name: str
    family: str
    sign: str = "unconstrained"            # 'positive' | 'negative' | 'unconstrained'
    adstock_decay: Optional[float] = None  # set for media


def build_feature_specs(df: pd.DataFrame, media_decay: float = 0.5) -> list:
    """Construct the candidate predictor list with marketing-mix sign priors."""
    cols = set(df.columns)
    specs = []

    def add(name, family, sign, decay=None):
        if name in cols:
            specs.append(FeatureSpec(name, family, sign, decay))

    # Own price -> negative
    add("Price per Volume", "Price", "negative")
    # Competitive / category price -> positive (higher competitor price helps us)
    add("Total Category Price Per Volume", "Category Price", "positive")
    add("Category P Price Per Volume", "Category Price", "positive")

    # Own distribution -> positive
    add("ACV Weighted Distribution", "Distribution", "positive")
    add("Avg Weekly Items per Store Selling", "Distribution", "positive")
    add("Total Points of Distribution", "Distribution", "positive")

    # Own trade execution (reach/frequency) -> positive
    for c in ["Weighted Weeks Price Reductions Only", "Weighted Weeks Feature Only",
              "Weighted Weeks Display Only", "Weighted Weeks Special Pack Only",
              "Weighted Weeks Feature and Display"]:
        add(c, "Trade", "positive")

    # Media spend (adstocked) -> positive
    for c in df.columns:
        if isinstance(c, str) and c.endswith("_Spend"):
            add(c, "Media", "positive", decay=media_decay)

    # Seasonality / trend
    add("Seasonality_Index", "Seasonality", "positive")
    add("Trend", "Trend", "unconstrained")

    # Macro-economic -> unconstrained (no strong prior)
    for c in ["Unemployment_Rate", "Median_CPI", "CFNAI", "Gas_Price", "UMCSENT",
              "T5YIE", "PSAVERT", "PCEC96", "SNAP_Participants", "Fedfunds",
              "Unempclaims", "Retail_Sales"]:
        add(c, "Macro", "unconstrained")

    # Competitive distribution & trade -> negative
    for c in df.columns:
        if isinstance(c, str) and c.startswith("_C_Competitor"):
            specs.append(FeatureSpec(c, "Competitive", "negative"))
    return specs


def assemble_matrix(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """Build the design matrix, applying adstock to media columns."""
    out = {}
    for s in specs:
        x = df[s.name].astype(float).values
        if s.adstock_decay is not None:
            x = adstock(x, s.adstock_decay)
        out[s.name] = x
    return pd.DataFrame(out, index=df.index)


# 4. Automated variable selection

def prune_by_vif(X: pd.DataFrame, threshold: float = 10.0) -> list:
    """Iteratively drop the highest-VIF predictor until all are below threshold."""
    keep = list(X.columns)
    while len(keep) > 1:
        Xc = sm.add_constant(X[keep])
        vifs = {keep[i]: variance_inflation_factor(Xc.values, i + 1)
                for i in range(len(keep))}
        worst, val = max(vifs.items(), key=lambda kv: kv[1])
        if val > threshold:
            keep.remove(worst)
        else:
            break
    return keep


def forward_stepwise(X: pd.DataFrame, y: pd.Series, p_enter: float = 0.05) -> list:
    """Forward selection by lowest p-value, keeping additions significant."""
    remaining = list(X.columns)
    selected = []
    while remaining:
        best_p, best_c = 1.0, None
        for c in remaining:
            cols = selected + [c]
            model = sm.OLS(y, sm.add_constant(X[cols])).fit()
            p = model.pvalues.get(c, 1.0)
            if p < best_p:
                best_p, best_c = p, c
        if best_c is not None and best_p < p_enter:
            selected.append(best_c)
            remaining.remove(best_c)
        else:
            break
    return selected


# 5. Constrained final fit

def _bounds_from_signs(specs_by_name: dict, cols: list):
    """Translate sign priors into (lower, upper) bounds for each coefficient.
    Intercept (first column) is always unconstrained."""
    lo, hi = [-np.inf], [np.inf]  # intercept
    for c in cols:
        sign = specs_by_name[c].sign
        if sign == "positive":
            lo.append(0.0); hi.append(np.inf)
        elif sign == "negative":
            lo.append(-np.inf); hi.append(0.0)
        else:
            lo.append(-np.inf); hi.append(np.inf)
    return np.array(lo), np.array(hi)


@dataclass
class FitResult:
    cols: list
    coef: pd.Series              # includes 'const'
    fitted: np.ndarray
    resid: np.ndarray
    r2: float
    adj_r2: float
    mape: float
    contributions: pd.DataFrame  # per-row additive due-tos
    tstats: pd.Series            # from unconstrained OLS (inference reference)
    vif: pd.Series
    meta: dict = field(default_factory=dict)


def constrained_fit(X: pd.DataFrame, y: pd.Series, specs_by_name: dict) -> FitResult:
    """Final model: bounded least squares enforcing marketing-mix sign priors."""
    cols = list(X.columns)
    Xd = np.column_stack([np.ones(len(X)), X.values])
    lo, hi = _bounds_from_signs(specs_by_name, cols)

    res = lsq_linear(Xd, y.values, bounds=(lo, hi), method="trf",
                     max_iter=5000, tol=1e-10)
    beta = res.x
    fitted = Xd @ beta
    resid = y.values - fitted

    n, k = len(y), len(cols)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y.values - y.values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    mape = float(np.mean(np.abs(resid / y.values))) * 100

    coef = pd.Series(beta, index=["const"] + cols)

    # Additive contributions (due-tos): const + sum_i beta_i * x_i == fitted
    contrib = pd.DataFrame({c: coef[c] * X[c].values for c in cols}, index=X.index)
    contrib.insert(0, "Base (const)", coef["const"])

    # Unconstrained OLS purely for t-stats / inference reference
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    tstats = ols.tvalues

    Xc = sm.add_constant(X)
    vif = pd.Series({c: variance_inflation_factor(Xc.values, i + 1)
                     for i, c in enumerate(cols)})

    return FitResult(cols, coef, fitted, resid, r2, adj_r2, mape,
                     contrib, tstats, vif,
                     meta={"durbin_watson": float(sm.stats.durbin_watson(resid))})


# 6. End-to-end driver for one slice

def run_slice(path: str, brand_sheet: str, channel: str,
              media_decay: float = 0.5, vif_threshold: float = 10.0,
              p_enter: float = 0.05, holdout_weeks: int = 26) -> dict:
    """Full Phase 1->2 pipeline for a single Brand x Channel slice."""
    df = load_slice(path, brand_sheet, channel)
    specs = build_feature_specs(df, media_decay=media_decay)
    specs_by_name = {s.name: s for s in specs}
    X_all = assemble_matrix(df, specs)
    y = df[TARGET].astype(float)

    # standardize for selection only
    Xz = (X_all - X_all.mean()) / X_all.std(ddof=0)
    Xz = Xz.loc[:, Xz.std() > 0]

    vif_keep = prune_by_vif(Xz, threshold=vif_threshold)
    selected = forward_stepwise(Xz[vif_keep], y, p_enter=p_enter)
    if not selected:
        selected = vif_keep[:5]

    # Final constrained fit in ORIGINAL units, then prune predictors whose
    # constrained coefficient was driven to the zero bound (no contribution)
    # and refit, so the delivered model contains only contributing drivers.
    sd = X_all[selected].std(ddof=0)
    for _ in range(len(selected)):
        fit = constrained_fit(X_all[selected], y, specs_by_name)
        impact = {c: abs(fit.coef[c]) * sd[c] for c in selected}
        scale = max(impact.values()) or 1.0
        dead = [c for c in selected if impact[c] / scale < 1e-4]
        if not dead:
            break
        selected = [c for c in selected if c not in dead]
    fit = constrained_fit(X_all[selected], y, specs_by_name)

    # sign-conflict diagnostic: where unconstrained OLS wants the opposite sign
    sign_conflicts = []
    for c in selected:
        prior = specs_by_name[c].sign
        t = fit.tstats.get(c, np.nan)
        if prior == "positive" and t < -1.96:
            sign_conflicts.append((c, prior, "data says negative", float(t)))
        elif prior == "negative" and t > 1.96:
            sign_conflicts.append((c, prior, "data says positive", float(t)))

    # time-based holdout validation: refit on the first weeks, score the tail
    split = len(df) - holdout_weeks
    fit_tr = constrained_fit(X_all[selected].iloc[:split], y.iloc[:split], specs_by_name)
    Xte = np.column_stack([np.ones(holdout_weeks), X_all[selected].iloc[split:].values])
    pred_te = Xte @ fit_tr.coef.values
    yte = y.iloc[split:].values
    holdout_mape = float(np.mean(np.abs((yte - pred_te) / yte))) * 100

    return {
        "df": df, "specs": specs, "specs_by_name": specs_by_name,
        "X_all": X_all, "y": y, "selected": selected, "fit": fit,
        "holdout_mape": holdout_mape, "split": split,
        "pred_te": pred_te, "yte": yte, "sign_conflicts": sign_conflicts,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Anonymized Data for Project.xlsx"
    r = run_slice(path, "Brand 1", "Channel 1")
    f = r["fit"]
    print(f"Selected {len(r['selected'])} predictors")
    print(f"R2={f.r2:.3f}  adjR2={f.adj_r2:.3f}  in-sample MAPE={f.mape:.1f}%  "
          f"holdout MAPE={r['holdout_mape']:.1f}%")
    print(f.coef.round(3))

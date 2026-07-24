"""
Big Chalk Capstone - Automated Regression Engine (core pipeline)
================================================================

Reusable, modular building blocks for modeling a weekly target (default
`Volume Sales`) for a single Brand x Channel slice, designed to scale to every
Brand x Channel combination in Phase 3 and to plug into the Phase 4 dashboard.

Design principles
-----------------
1. No target leakage. Columns that are algebraic decompositions of the target
   ("Volume Sales <merch condition>", "Dollar Sales ...") are EXCLUDED as
   predictors. Kept only for cross-checks, never as features.

2. Config-driven marketing-mix framing (Alex, sponsor review 2026-07-06).
   Predictors are NOT hard-coded by name. A variable-config table (CSV) maps
   each raw column to a family, an expected SIGN (positive / negative /
   unconstrained), optional custom coefficient bounds (e.g. 0.10-0.25), a
   per-variable adstock decay, and a role:
       auto    - goes through automated selection (default)
       force   - always in the model (client-mandated variables)
       exclude - never in the model
   `generate_variable_config()` writes a starter template from heuristics;
   the analyst edits the CSV, not the code.

3. Two-stage estimation:
       (a) automated variable selection (VIF prune -> forward stepwise) on
           standardized data to choose a parsimonious, low-collinearity set;
       (b) FINAL fit by bounded/sign-constrained least squares (scipy
           lsq_linear, TRF) in original units so coefficients are
           interpretable and contributions ("due-tos") decompose additively.

   Why these methods (for the write-up):
       - lsq_linear over Ridge: Ridge cannot honor sign/box constraints;
         bounded least squares can enforce any [lo, hi] per coefficient.
       - forward stepwise over Lasso/tree importance: transparent, each
         addition is a testable p-value decision we can explain to the client.
       - VIF prune over pairwise correlation/PCA: catches a variable that is
         redundant with a *combination* of others, and keeps raw (explainable)
         variables unlike PCA.
       - time-based holdout over random K-fold: weekly series are
         autocorrelated; random folds leak the future into training.

4. Adstock (normalized geometric, Big Chalk convention):
       a_t = (1 - decay) * x_t + decay * a_{t-1}
   The (1-decay) share of this week's execution hits this week; the remainder
   carries forward. Total adstocked impressions ~= total raw impressions over
   a long window (the naive a_t = x_t + decay*a_{t-1} inflates totals).
   Decay is customizable per media variable (search-type media ~0.2 ... TV/CTV
   ~0.7; 0.5 is the industry-standard default).

5. Set-year modeling window. Model on set years (52/104/156 weeks; default the
   latest 104). A validation model always reserves the last up to 13 weeks as a
   time-based holdout (independent of the window); the reported model refits the
   same structure on the full window. Contributions are reported both weekly
   and summed by model year for year-over-year comparison.

Author: Capstone team (Feifan, Boqi, Jiahao) | June-July 2026
"""

from __future__ import annotations
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

# Serializes ALL config-file mutations within a process (shared by the
# dashboard's save/upload/regenerate/reset and default creation here). RLock so
# a thread already holding it can re-enter. Process-local; cross-process needs
# OS file locks.
_CFG_WRITE_LOCK = threading.RLock()

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Transforms
# ---------------------------------------------------------------------------

def adstock(x: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Normalized geometric adstock (Big Chalk convention, Alex 2026-07-06):

        a_t = (1 - decay) * x_t + decay * a_{t-1}

    (1-decay) of this week's execution lands this week; the remainder decays
    in over following weeks, so sum(adstocked) ~= sum(raw) over a long window
    and adstocked impressions never systematically exceed raw impressions.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        carry = (1.0 - decay) * v + decay * carry
        out[i] = carry
    return out


def adstock_totals_check(x: np.ndarray, decay: float = 0.5) -> dict:
    """Sanity check: total adstocked vs total raw (should be ~1.0; slightly
    below because the tail of the decay extends past the data window)."""
    raw = float(np.sum(x))
    dec = float(np.sum(adstock(x, decay)))
    return {"raw_total": raw, "adstocked_total": dec,
            "ratio": dec / raw if raw else np.nan}


# ---------------------------------------------------------------------------
# 3. Predictor specification (config-table driven)
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """One candidate predictor: source column, family, expected sign,
    optional custom coefficient bounds and per-variable adstock decay."""
    name: str
    family: str
    sign: str = "unconstrained"            # 'positive' | 'negative' | 'unconstrained'
    adstock_decay: Optional[float] = None  # set for media
    coef_lower: Optional[float] = None     # custom bound overrides sign
    coef_upper: Optional[float] = None
    role: str = "auto"                     # 'auto' | 'force' | 'exclude'


@dataclass
class ModelConfig:
    """Run-level knobs. Everything the sponsor asked to be adjustable."""
    target: str = "Volume Sales"      # any raw column can be the dependent
    model_weeks: int = 104            # set-year window (52 / 104 / 156)
    holdout_weeks: Optional[int] = None  # None -> ALWAYS reserve a validation
    #                                      tail of up to 13 wks (independent of
    #                                      model_weeks; floored to keep >=5
    #                                      training wks). Set 0 to disable.
    vif_threshold: float = 10.0
    p_enter: float = 0.05
    default_media_decay: float = 0.5
    force_include: list = field(default_factory=list)   # client-mandated
    exclude: list = field(default_factory=list)         # never model
    variable_config: Optional[str] = None  # path to variable_config.csv


# ─────────────────────────────────────────────────────────────────────────
# Two-tier variable-config resolution (SINGLE SOURCE OF TRUTH for the app AND
# the CLI runners). Product default + optional Product × Channel override;
# override wins. Naming/paths must match what the dashboard reads and writes.
# ─────────────────────────────────────────────────────────────────────────

def _cfg_slug(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def config_dir() -> str:
    return os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..")), "configs")


def default_config_path(data_path: str, sheet: str) -> str:
    """Product DEFAULT config (channel = ALL)."""
    ds = _cfg_slug(os.path.splitext(os.path.basename(data_path))[0])[:24]
    return os.path.join(config_dir(), f"varconfig_{ds}_{_cfg_slug(sheet)}.csv")


def override_config_path(data_path: str, sheet: str, channel: str) -> str:
    """Product × Channel OVERRIDE config."""
    ds = _cfg_slug(os.path.splitext(os.path.basename(data_path))[0])[:24]
    return os.path.join(
        config_dir(), f"varconfig_{ds}_{_cfg_slug(sheet)}__ch_"
        f"{_cfg_slug(channel)}.csv")


def resolve_config_path(data_path: str, sheet: str,
                        channel: str) -> Optional[str]:
    """Runtime resolution shared by dashboard + CLI: channel override if it
    exists, else the product default if it exists, else None (caller then
    falls back to auto-generation). Override > default."""
    ov = override_config_path(data_path, sheet, channel)
    if os.path.exists(ov):
        return ov
    dp = default_config_path(data_path, sheet)
    if os.path.exists(dp):
        return dp
    return None


# ACV force-include is baked into the generated DEFAULT config's ROLE (one
# place), so every entry point that uses a config file honors it — and honors
# the analyst overriding it back to `auto`. No caller should also pass
# force_include=["ACV …"] on top of a config file, or the two disagree.
def generate_default_config(df: pd.DataFrame) -> pd.DataFrame:
    cfg = generate_variable_config(df)
    cfg.loc[cfg["variable"] == "ACV Weighted Distribution", "role"] = "force"
    return cfg


def load_or_create_default_config(data_path: str, sheet: str, df=None,
                                  regenerate: bool = False) -> str:
    """Return the Product-default config path, creating it (auto-generated,
    ACV force-included via role) if missing. Single creation path for app +
    CLI so a fresh dataset gets identical defaults everywhere."""
    p = default_config_path(data_path, sheet)
    if os.path.exists(p) and not regenerate:
        return p
    with _CFG_WRITE_LOCK:
        if os.path.exists(p) and not regenerate:   # re-check under lock
            return p
        if df is None:
            df = pd.read_excel(data_path, sheet_name=sheet)
        os.makedirs(config_dir(), exist_ok=True)
        generate_default_config(df).to_csv(p, index=False)
    return p


# Anything that algebraically decomposes the *default* target. When the target
# changes, decompositions of the new target are excluded dynamically.
LEAKAGE_PREFIXES = ("Volume Sales", "Dollar Sales")
TARGET = "Volume Sales"   # backward-compatible module default

# Suggested starting decays by media type (Alex: search ~0.2, TV/CTV long ~0.7)
_DECAY_HINTS = (("google", 0.2), ("search", 0.2),
                ("ctv", 0.7), ("tv", 0.7), ("video", 0.7), ("radio", 0.6))


def _suggest_decay(col: str, default: float) -> float:
    low = col.lower()
    for pat, d in _DECAY_HINTS:
        if pat in low:
            return d
    return default


def generate_variable_config(df: pd.DataFrame, path: Optional[str] = None,
                             default_media_decay: float = 0.5) -> pd.DataFrame:
    """Build a starter variable-config table from naming heuristics and
    optionally write it to CSV. THE CSV, NOT THIS CODE, is the source of
    truth once the analyst edits it - rename/add columns freely there."""
    rows = []

    def add(name, family, sign, decay=None, lo=None, hi=None, role="auto"):
        if name in df.columns:
            rows.append({"variable": name, "family": family, "sign": sign,
                         "adstock_decay": decay, "coef_lower": lo,
                         "coef_upper": hi, "role": role})

    add("Price per Volume", "Price", "negative")
    add("Total Category Price Per Volume", "Category Price", "positive")
    add("Category P Price Per Volume", "Category Price", "positive")

    add("ACV Weighted Distribution", "Distribution", "positive")
    add("Avg Weekly Items per Store Selling", "Distribution", "positive")
    add("Total Points of Distribution", "Distribution", "positive")

    for c in ["Weighted Weeks Price Reductions Only", "Weighted Weeks Feature Only",
              "Weighted Weeks Display Only", "Weighted Weeks Special Pack Only",
              "Weighted Weeks Feature and Display"]:
        add(c, "Trade", "positive")

    for c in df.columns:
        if isinstance(c, str) and c.endswith("_Spend"):
            add(c, "Media", "positive",
                decay=_suggest_decay(c, default_media_decay))

    add("Seasonality_Index", "Seasonality", "positive")
    add("Trend", "Trend", "unconstrained")

    for c in ["Unemployment_Rate", "Median_CPI", "CFNAI", "Gas_Price", "UMCSENT",
              "T5YIE", "PSAVERT", "PCEC96", "SNAP_Participants", "Fedfunds",
              "Unempclaims", "Retail_Sales"]:
        add(c, "Macro", "unconstrained")

    for c in df.columns:
        if isinstance(c, str) and c.startswith("_C_"):
            # competitive price would be positive; competitive distribution
            # and trade pressure pull our volume down -> negative
            add(c, "Competitive", "negative")

    cfg = pd.DataFrame(rows)
    if path:
        cfg.to_csv(path, index=False)
    return cfg


def load_variable_config(cfg: Union[str, pd.DataFrame],
                         df: Optional[pd.DataFrame] = None) -> list:
    """Turn a variable-config table (CSV path or DataFrame) into FeatureSpecs.
    Rows whose variable is missing from `df` (if given) are skipped, so one
    master config can serve data sets with different column subsets."""
    tbl = pd.read_csv(cfg) if isinstance(cfg, str) else cfg.copy()
    specs = []
    for _, r in tbl.iterrows():
        name = str(r["variable"])
        if df is not None and name not in df.columns:
            continue
        def _f(key):
            v = r.get(key)
            return None if pd.isna(v) else float(v)
        specs.append(FeatureSpec(
            name=name,
            family=str(r.get("family", "Other")),
            sign=str(r.get("sign", "unconstrained")).strip().lower(),
            adstock_decay=_f("adstock_decay"),
            coef_lower=_f("coef_lower"),
            coef_upper=_f("coef_upper"),
            role=str(r.get("role", "auto") or "auto").strip().lower(),
        ))
    return specs


def build_feature_specs(df: pd.DataFrame, media_decay: float = 0.5,
                        config: Union[str, pd.DataFrame, None] = None) -> list:
    """Candidate predictor list. Uses the variable-config table when given;
    falls back to generated heuristics otherwise (same result as a fresh
    template)."""
    if config is not None:
        return load_variable_config(config, df)
    return load_variable_config(
        generate_variable_config(df, default_media_decay=media_decay), df)


def assemble_matrix(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """Build the design matrix, applying per-variable adstock to media."""
    out = {}
    for s in specs:
        x = df[s.name].astype(float).values
        if s.adstock_decay is not None:
            x = adstock(x, s.adstock_decay)
        out[s.name] = x
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# 4. Automated variable selection
# ---------------------------------------------------------------------------

def _vif_all(X: pd.DataFrame) -> pd.Series:
    """All VIFs at once via the diagonal of the inverse correlation matrix.
    Identity: VIF_i = [corr(X)^-1]_ii  (equivalent to the k auxiliary
    regressions statsmodels runs, ~100x faster for wide X)."""
    R = np.corrcoef(X.values, rowvar=False)
    R = np.atleast_2d(R)
    try:
        inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(R)
    return pd.Series(np.diag(inv), index=X.columns)


def prune_by_vif(X: pd.DataFrame, threshold: float = 10.0,
                 protect: Optional[list] = None) -> list:
    """Iteratively drop the highest-VIF predictor until all are below
    threshold. `protect`ed (force-include) variables are never dropped -
    the client paid for them, they stay (VIF gets thrown out the window)."""
    protect = set(protect or [])
    keep = list(X.columns)
    while len(keep) > 1:
        vifs = _vif_all(X[keep])
        droppable = vifs.drop([c for c in keep if c in protect])
        if droppable.empty:
            break
        worst = droppable.idxmax()
        if droppable[worst] > threshold:
            keep.remove(worst)
        else:
            break
    return keep


def _pvalue_of_last(Xd: np.ndarray, y: np.ndarray) -> float:
    """p-value of the LAST column's coefficient in OLS (Xd includes the
    intercept column). Direct normal-equations solve - same numbers as
    statsmodels OLS, without the per-call SVD/wrapper overhead."""
    from scipy import stats as _st
    n, k = Xd.shape
    dof = n - k
    if dof <= 0:
        return 1.0
    XtX = Xd.T @ Xd
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return 1.0
    beta = XtX_inv @ (Xd.T @ y)
    resid = y - Xd @ beta
    s2 = float(resid @ resid) / dof
    var = s2 * XtX_inv[-1, -1]
    if var <= 0:
        return 1.0
    t = beta[-1] / np.sqrt(var)
    return 2.0 * float(_st.t.sf(abs(t), dof))


def forward_stepwise(X: pd.DataFrame, y: pd.Series, p_enter: float = 0.05,
                     start_with: Optional[list] = None) -> list:
    """Forward selection by lowest p-value, keeping additions significant.
    `start_with` (force-include) variables are seeded into the model."""
    selected = [c for c in (start_with or []) if c in X.columns]
    remaining = [c for c in X.columns if c not in selected]
    yv = y.values.astype(float)
    ones = np.ones((len(X), 1))
    while remaining:
        base = X[selected].values if selected else np.empty((len(X), 0))
        best_p, best_c = 1.0, None
        for c in remaining:
            Xd = np.column_stack([ones, base, X[c].values])
            p = _pvalue_of_last(Xd, yv)
            if p < best_p:
                best_p, best_c = p, c
        if best_c is not None and best_p < p_enter:
            selected.append(best_c)
            remaining.remove(best_c)
        else:
            break
    return selected


# ---------------------------------------------------------------------------
# 5. Constrained final fit
# ---------------------------------------------------------------------------

def _bounds_from_specs(specs_by_name: dict, cols: list):
    """(lower, upper) bounds per coefficient. Custom bounds in the config
    override the sign default; intercept is always unconstrained."""
    lo, hi = [-np.inf], [np.inf]  # intercept
    for c in cols:
        s = specs_by_name[c]
        if s.coef_lower is not None or s.coef_upper is not None:
            lo.append(s.coef_lower if s.coef_lower is not None else -np.inf)
            hi.append(s.coef_upper if s.coef_upper is not None else np.inf)
        elif s.sign == "positive":
            lo.append(0.0); hi.append(np.inf)
        elif s.sign == "negative":
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
    contributions: pd.DataFrame  # per-row additive due-tos (intercept separate)
    tstats: pd.Series            # from unconstrained OLS (inference reference)
    vif: pd.Series
    meta: dict = field(default_factory=dict)


def constrained_fit(X: pd.DataFrame, y: pd.Series, specs_by_name: dict) -> FitResult:
    """Final model: bounded least squares enforcing sign/box constraints."""
    cols = list(X.columns)
    Xd = np.column_stack([np.ones(len(X)), X.values])
    lo, hi = _bounds_from_specs(specs_by_name, cols)

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
    nz = y.values != 0   # MAPE over non-zero actuals only (zero-sales weeks
    #                        would divide by zero in partial-distribution channels)
    mape = float(np.mean(np.abs(resid[nz] / y.values[nz]))) * 100 if nz.any() else np.nan

    coef = pd.Series(beta, index=["const"] + cols)

    # Additive contributions (due-tos): intercept its own line (Alex), and
    # const + sum_i beta_i * x_i == fitted exactly.
    contrib = pd.DataFrame({c: coef[c] * X[c].values for c in cols}, index=X.index)
    contrib.insert(0, "Intercept", coef["const"])

    # Unconstrained OLS purely for t-stats / inference reference
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    tstats = ols.tvalues

    Xc = sm.add_constant(X)
    vif = pd.Series({c: variance_inflation_factor(Xc.values, i + 1)
                     for i, c in enumerate(cols)})

    return FitResult(cols, coef, fitted, resid, r2, adj_r2, mape,
                     contrib, tstats, vif,
                     meta={"durbin_watson": float(sm.stats.durbin_watson(resid))})


# ---------------------------------------------------------------------------
# 6. Contribution reporting
# ---------------------------------------------------------------------------

def assign_model_years(dates: pd.Series, weeks_per_year: int = 52) -> pd.Series:
    """Label consecutive 52-week blocks counting back from the latest week:
    latest block = highest year number (the 'current' model year)."""
    n = len(dates)
    idx = np.arange(n)
    blocks_back = (n - 1 - idx) // weeks_per_year      # 0 = most recent block
    n_years = int(blocks_back.max()) + 1
    year_no = n_years - blocks_back                    # 1 = oldest
    out = list(year_no)
    s = pd.Series(out, index=dates.index, name="model_year")
    # human-readable label per year
    lab = {}
    for yn in sorted(set(out)):
        d = dates[s == yn]
        lab[yn] = f"Year {yn} ({d.iloc[0]:%b %Y}-{d.iloc[-1]:%b %Y})"
    return s.map(lab)


def contributions_by_year(fit: FitResult, dates: pd.Series) -> pd.DataFrame:
    """Signed due-to sums per driver per model year, plus YoY change - the
    'how much better/worse am I this year, and due to what' view."""
    years = assign_model_years(dates.reset_index(drop=True))
    tbl = fit.contributions.copy()
    tbl.index = years.values
    out = tbl.groupby(level=0).sum().T          # rows = drivers, cols = years
    cols = list(out.columns)
    if len(cols) >= 2:
        out["YoY_change"] = out[cols[-1]] - out[cols[-2]]
        prev = out[cols[-2]].replace(0, np.nan)
        out["YoY_pct"] = (out["YoY_change"] / prev.abs()) * 100
    return out


def avg_weekly_contributions(fit: FitResult, X_raw: pd.DataFrame,
                             specs_by_name: dict) -> pd.DataFrame:
    """Average weekly signed contribution per driver. For media (anything
    with an adstock decay), average ONLY over weeks with nonzero raw
    execution - never average over the zeros of a flight that started
    mid-window (Alex)."""
    rows = []
    for c in fit.cols:
        contrib = fit.contributions[c]
        spec = specs_by_name.get(c)
        if spec is not None and spec.adstock_decay is not None and c in X_raw:
            mask = X_raw[c].astype(float).values > 0
            n_active = int(mask.sum())
            avg = float(contrib.values[mask].mean()) if n_active else 0.0
        else:
            n_active = len(contrib)
            avg = float(contrib.mean())
        rows.append({"driver": c, "avg_weekly_contribution": avg,
                     "weeks_with_execution": n_active})
    rows.append({"driver": "Intercept",
                 "avg_weekly_contribution": float(fit.contributions["Intercept"].mean()),
                 "weeks_with_execution": len(fit.contributions)})
    return pd.DataFrame(rows).set_index("driver")


# ---------------------------------------------------------------------------
# 7. End-to-end driver for one slice
# ---------------------------------------------------------------------------

def run_slice(path: str, brand_sheet: str, channel: str,
              config: Optional[ModelConfig] = None,
              # legacy kwargs kept for backward compatibility
              media_decay: Optional[float] = None,
              vif_threshold: Optional[float] = None,
              p_enter: Optional[float] = None,
              holdout_weeks: Optional[int] = None,
              df: Optional[pd.DataFrame] = None) -> dict:
    """Full pipeline for a single Brand x Channel slice, config-driven.
    Pass `df` (a full brand-sheet DataFrame) to skip re-reading the Excel
    file — used by the Phase 3 batch runner."""
    cfg = config or ModelConfig()
    if media_decay is not None: cfg.default_media_decay = media_decay
    if vif_threshold is not None: cfg.vif_threshold = vif_threshold
    if p_enter is not None: cfg.p_enter = p_enter
    if holdout_weeks is not None: cfg.holdout_weeks = holdout_weeks

    if df is not None:
        df_full = df[df["Geography"] == channel].copy()
        df_full["date"] = df_full["Time"].apply(_parse_week)
        df_full = df_full.sort_values("date").reset_index(drop=True)
    else:
        df_full = load_slice(path, brand_sheet, channel)
    if cfg.target not in df_full.columns:
        raise ValueError(f"Target '{cfg.target}' not in data")

    # --- window + ALWAYS-reserved validation tail (Jerry / M1) ---
    # ONE structure, two fits:
    #   The VALIDATION model SELECTS the variable structure on the window ending
    #   BEFORE an always-reserved tail (up to 13 wks) — selection never sees the
    #   tail (no leakage) — and is scored on that tail => the out-of-sample
    #   holdout MAPE. The REPORTED model refits the SAME structure on the full
    #   window ending at the latest week (all data incl. the tail) => the
    #   coefficients/due-tos we display. So the holdout validates the SAME
    #   structure and modeling process out-of-sample (the reported coefficients
    #   are that structure refit on all data — the holdout is not a check of
    #   those exact refit values), and the reported metric is a true
    #   out-of-sample number (never the all-data in-sample fit relabeled).
    n = len(df_full)
    mw = int(cfg.model_weeks)
    cap = cfg.holdout_weeks if cfg.holdout_weeks is not None else 13
    val_hold = int(min(max(cap, 0), max(n - 1, 0)))
    if n - val_hold < 5:                        # keep ≥5 training weeks
        val_hold = max(0, n - 5)

    # candidate predictors (config-driven; identical across windows)
    specs = build_feature_specs(df_full, media_decay=cfg.default_media_decay,
                                config=cfg.variable_config)
    leak = tuple({cfg.target, "Dollar Sales", "Volume Sales"}
                 if cfg.target.startswith(("Volume Sales", "Dollar Sales"))
                 else {cfg.target})
    specs = [s for s in specs
             if s.name != cfg.target and not s.name.startswith(leak)]
    force = [s.name for s in specs if s.role == "force"] + \
            [c for c in cfg.force_include if c in df_full.columns]
    drop = {s.name for s in specs if s.role == "exclude"} | set(cfg.exclude)
    specs = [s for s in specs if s.name not in drop]
    specs_by_name = {s.name: s for s in specs}
    force = [c for c in dict.fromkeys(force) if c in specs_by_name]
    spec_names = [s.name for s in specs]

    def _assemble(a, b):
        dfx = df_full.iloc[a:b].reset_index(drop=True)
        return (dfx, assemble_matrix(dfx, specs), dfx[spec_names].copy(),
                dfx[cfg.target].astype(float))

    def _select(Xtr, ytr):
        """VIF prune + forward stepwise + dead-coef prune -> selected names."""
        Xz = (Xtr - Xtr.mean()) / Xtr.std(ddof=0)
        Xz = Xz.loc[:, Xz.std() > 0]
        favail = [c for c in force if c in Xz.columns]
        vif_keep = prune_by_vif(Xz, threshold=cfg.vif_threshold, protect=favail)
        selected = forward_stepwise(Xz[vif_keep], ytr, p_enter=cfg.p_enter,
                                    start_with=favail) or vif_keep[:5]
        sd = Xtr[selected].std(ddof=0)
        for _ in range(len(selected)):
            f = constrained_fit(Xtr[selected], ytr, specs_by_name)
            impact = {c: abs(f.coef[c]) * sd[c] for c in selected}
            scale = max(impact.values()) or 1.0
            dead = [c for c in selected
                    if impact[c] / scale < 1e-4 and c not in favail]
            if not dead:
                break
            selected = [c for c in selected if c not in dead]
        return selected

    # REPORTED window (full history-length ending at the latest week)
    df, X_all, X_raw, y = _assemble(max(n - mw, 0), n)

    # VALIDATION: SELECT on training (no tail leakage) + score the reserved tail
    holdout_mape = np.nan
    pred_te = yte = np.array([])
    if val_hold > 0:
        val_end = n - val_hold
        vdf, vX, _, vy = _assemble(max(val_end - mw, 0), n)
        sv = len(vdf) - val_hold
        selected = _select(vX.iloc[:sv], vy.iloc[:sv])
        vfit = constrained_fit(vX[selected].iloc[:sv], vy.iloc[:sv], specs_by_name)
        Xte = np.column_stack([np.ones(val_hold),
                               vX[selected].iloc[sv:].values])
        pred_te = Xte @ vfit.coef.values
        yte = vy.iloc[sv:].values
        nz = yte != 0
        holdout_mape = float(np.mean(np.abs((yte[nz] - pred_te[nz]) / yte[nz]))) * 100 \
            if nz.any() else np.nan
    else:                                       # no tail -> select on reported
        selected = _select(X_all, y)

    # REPORTED model — the SAME structure, refit on the full reported window
    fit = constrained_fit(X_all[selected], y, specs_by_name)
    force_avail = [c for c in force if c in selected]
    split = len(df) - val_hold                  # where the reserved tail begins

    adstock_checks = {s.name: adstock_totals_check(df[s.name].values, s.adstock_decay)
                      for s in specs if s.adstock_decay is not None}

    # sign-conflict diagnostic on the REPORTED fit
    sign_conflicts = []
    for c in selected:
        prior = specs_by_name[c].sign
        t = fit.tstats.get(c, np.nan)
        if prior == "positive" and t < -1.96:
            sign_conflicts.append((c, prior, "data says negative", float(t)))
        elif prior == "negative" and t > 1.96:
            sign_conflicts.append((c, prior, "data says positive", float(t)))

    # contributions over the FULL reported window (all data incl. former tail)
    contrib_by_year = contributions_by_year(fit, df["date"])
    avg_contrib = avg_weekly_contributions(fit, X_raw, specs_by_name)

    return {
        "df": df, "config": cfg, "specs": specs, "specs_by_name": specs_by_name,
        "X_all": X_all, "X_raw": X_raw, "y": y, "selected": selected,
        "forced": force_avail, "fit": fit,
        "holdout_mape": holdout_mape, "split": split,
        "pred_te": pred_te, "yte": yte, "sign_conflicts": sign_conflicts,
        "adstock_checks": adstock_checks, "val_hold": val_hold,
        "contrib_by_year": contrib_by_year, "avg_contrib": avg_contrib,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Anonymized Data for Project.xlsx"
    r = run_slice(path, "Brand 1", "Channel 1")
    f = r["fit"]
    print(f"Selected {len(r['selected'])} predictors "
          f"(window={r['config'].model_weeks}w, holdout={len(r['yte'])}w)")
    print(f"R2={f.r2:.3f}  adjR2={f.adj_r2:.3f}  in-sample MAPE={f.mape:.1f}%  "
          f"holdout MAPE={r['holdout_mape']:.1f}%")
    print(f.coef.round(3))

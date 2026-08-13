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


# ─────────────────────────────────────────────────────────────────────────
# 1b. FIXED calendar window (Alex, 2026-07-27)
#
# The window used to be "the last `model_weeks` ROWS of whatever this slice
# happens to have" — data-dependent. A slice that was delisted in Jun 2023
# (Brand 1/2 x Channel 8: 25 weeks, all of them 18 months before the modeled
# period) therefore still produced a model, and its due-tos leaked into the
# exports as a disjoint chunk of history with no neighbours.
#
# Now the window is an ABSOLUTE date range, anchored to the LATEST week in the
# DATASET (not in the slice) and identical for every Brand x Channel:
#     [anchor - (model_weeks-1) weeks  ..  anchor]
# The data is dependent on the window, not the window on the data. A slice with
# too little data INSIDE that range is not modeled at all — it raises
# InsufficientWindowData and is reported as skipped, with the reason.
# ─────────────────────────────────────────────────────────────────────────

class InsufficientWindowData(ValueError):
    """Raised when a slice does not have enough data inside the fixed window
    to be worth modeling. Carries the counts so callers can report WHY."""

    def __init__(self, message: str, *, weeks: int = 0, nonzero: int = 0,
                 window_start=None, window_end=None):
        super().__init__(message)
        self.weeks = weeks
        self.nonzero = nonzero
        self.window_start = window_start
        self.window_end = window_end


_ANCHOR_CACHE: dict = {}


def dataset_anchor_week(path: str, sheets: Optional[list] = None):
    """The latest week present ANYWHERE in the workbook — the common anchor
    every slice's window ends on. Cached per (path, mtime, size) so repeated
    runs don't re-read the workbook."""
    key = None
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime, st.st_size)
        if key in _ANCHOR_CACHE:
            return _ANCHOR_CACHE[key]
    except OSError:
        pass
    xl = pd.ExcelFile(path)
    names = sheets or [s for s in xl.sheet_names if s.lower().startswith("brand")]
    latest = None
    for s in names:
        d = xl.parse(s, usecols=["Time"])["Time"].apply(_parse_week).max()
        if d is not None and (latest is None or d > latest):
            latest = d
    latest = pd.Timestamp(latest) if latest is not None else None
    if key is not None:
        _ANCHOR_CACHE[key] = latest
    return latest


def resolve_window(anchor, model_weeks: int):
    """(start, end) timestamps for a `model_weeks`-long window ending on the
    anchor week, inclusive of both ends."""
    end = pd.Timestamp(anchor)
    start = end - pd.Timedelta(weeks=int(model_weeks) - 1)
    return start, end


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


def hill(x: np.ndarray, midpoint: float, slope: float,
         ref: Optional[float] = None) -> np.ndarray:
    """HILL SATURATION — the second non-linearity of media (Alex 2026-07-27).

        H(x) = x^s / (x^s + k^s)          k = midpoint × ref

    Alex described this curve on the call and half-remembered its origin:
    "some Bill and this other data scientist created this for the efficacy of
    drugs… a curve that has a midpoint and a slope, so it makes it sigmoidal."
    It is the **Hill equation** — A. V. Hill, 1910, fitting oxygen binding to
    haemoglobin, later the standard dose–response curve in pharmacology, and
    now the saturation function in both Meta's Robyn and Google's Meridian.
    Same shape, same two parameters he named.

    Parameters
    ----------
    midpoint : the half-saturation point as a FRACTION of `ref` (γ ∈ (0, 1]).
        At x = γ·ref the response is exactly half of its maximum. Expressed
        as a fraction so the parameter transfers across brands and channels
        whose spend levels differ by orders of magnitude.
    slope : the shape s.
        s > 1 gives the S-curve Alex described — slow start ("the first 10
        impressions probably don't do a lot"), steep through the midpoint,
        then flattening. s < 1 gives a concave curve that diminishes straight
        from the origin, which aggregate weekly data often prefers.
    ref : the scale the midpoint is a fraction OF — the max of the (adstocked)
        series. MUST be computed on training weeks only and then reused, or
        the transform itself peeks at the holdout.

    Returns values in [0, 1): the coefficient on a saturated variable is
    therefore the **maximum weekly volume that variable can deliver**, and
    contributions stay additive, so the due-to decomposition is unchanged.
    """
    x = np.asarray(x, dtype=float)
    if ref is None:
        ref = float(np.nanmax(x)) if x.size else 0.0
    k = float(midpoint) * float(ref)
    if not np.isfinite(k) or k <= 0:
        return np.zeros_like(x)
    s = float(slope)
    # negatives can't be raised to a fractional power; media is non-negative
    # anyway, and clipping keeps a stray negative from producing a NaN column
    xs = np.power(np.clip(x, 0.0, None), s)
    ks = k ** s
    out = xs / (xs + ks)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def saturation_ref(x: np.ndarray) -> float:
    """The reference scale a Hill midpoint is a fraction of. Uses the max of
    the series it is given — callers pass TRAINING rows only."""
    x = np.asarray(x, dtype=float)
    m = float(np.nanmax(x)) if x.size else 0.0
    return m if np.isfinite(m) and m > 0 else 0.0


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
    optional custom coefficient bounds, per-variable adstock decay and
    reporting scale."""
    name: str
    family: str
    sign: str = "unconstrained"            # 'positive' | 'negative' | 'unconstrained'
    adstock_decay: Optional[float] = None  # set for media
    coef_lower: Optional[float] = None     # custom bound overrides sign
    coef_upper: Optional[float] = None
    role: str = "auto"                     # 'auto' | 'force' | 'exclude'
    # Divide the variable by this before modeling (Alex 2026-07-27: "really
    # big things we'll divide by 1000 or 10,000, usually with media
    # impressions… if you want to scale it, include that scale in the variable
    # tab"). The coefficient is then read "per 1,000 impressions" instead of
    # "per impression" — a units change, NOT a model change: β is divided and
    # x multiplied by the same constant, so every contribution, due-to and
    # fitted value is bit-for-bit identical. Custom bounds are stated in
    # ORIGINAL units and transformed with it.
    scale: float = 1.0
    # Hill saturation (F3). Both None = no saturation, i.e. the variable stays
    # linear — the historical behaviour and still the default.
    sat_midpoint: Optional[float] = None   # γ: half-saturation as a fraction
    sat_slope: Optional[float] = None      # s: shape (>1 S-curve, <1 concave)


@dataclass
class ModelConfig:
    """Run-level knobs. Everything the sponsor asked to be adjustable."""
    target: str = "Volume Sales"      # any raw column can be the dependent
    model_weeks: int = 104            # set-year window (52 / 104 / 156)
    # FIXED window (Alex 2026-07-27). `window_end` is the shared anchor week
    # every slice's window ends on — pass the DATASET-wide latest week
    # (cp.dataset_anchor_week) so all slices model the SAME calendar range.
    # None -> derived from the data handed to run_slice (whole brand sheet,
    # all channels) as a fallback; never from the single channel's own tail.
    window_end: Optional[object] = None
    # A slice needs at least this much data INSIDE the window to be modeled;
    # otherwise it is skipped with a reason instead of producing a phantom
    # model out of stale history.
    min_weeks: int = 52               # weeks of any data in the window
    min_nonzero_weeks: int = 26       # weeks with a non-zero target
    holdout_weeks: Optional[int] = None  # None -> ALWAYS reserve a validation
    #                                      tail of up to 13 wks (independent of
    #                                      model_weeks; floored to keep >=5
    #                                      training wks). Set 0 to disable.
    # ── selection hyperparameters (Alex 2026-07-27: "is there a way to be
    # more specific on our end with being overly punitive with your variable
    # selection, or really lax — is that a parameter that we can adjust?") ──
    # `max_per_family` caps how many variables of each family may enter, e.g.
    # {"Trade": 3, "Competitive": 2, "Macro": 2} — his "I want only two to
    # three trade, but I want all media". EMPTY BY DEFAULT: no caps, so
    # existing results are untouched until someone opts in.
    max_per_family: dict = field(default_factory=dict)
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


def _suggest_scale(col: str, df: pd.DataFrame) -> float:
    """Starting scale for a variable: a round power of 10 that brings a very
    large series into a readable range, so a media coefficient reads "per
    1,000 impressions" rather than per impression (Alex). Only ever a
    suggestion — the config CSV is the source of truth once edited, and the
    scale changes units only, never the fit."""
    try:
        m = float(pd.to_numeric(df[col], errors="coerce").abs().max())
    except Exception:
        return 1.0
    # deliberately conservative: only series big enough that the raw
    # coefficient is unreadable get a suggestion, so a fresh config doesn't
    # surprise anyone with unfamiliar units on ordinary variables
    if not np.isfinite(m):
        return 1.0
    if m >= 1e9:
        return 1e6
    if m >= 1e6:
        return 1e3
    return 1.0


def generate_variable_config(df: pd.DataFrame, path: Optional[str] = None,
                             default_media_decay: float = 0.5) -> pd.DataFrame:
    """Build a starter variable-config table from naming heuristics and
    optionally write it to CSV. THE CSV, NOT THIS CODE, is the source of
    truth once the analyst edits it - rename/add columns freely there."""
    rows = []

    def add(name, family, sign, decay=None, lo=None, hi=None, role="auto",
            scale=None):
        if name in df.columns:
            rows.append({"variable": name, "family": family, "sign": sign,
                         "adstock_decay": decay, "coef_lower": lo,
                         "coef_upper": hi, "role": role,
                         "scale": scale if scale is not None
                         else _suggest_scale(name, df),
                         # saturation off by default — the optimizer or the
                         # analyst turns it on per variable
                         "sat_midpoint": None, "sat_slope": None})

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
        sc = _f("scale")
        mid, slp = _f("sat_midpoint"), _f("sat_slope")
        if mid is None or slp is None:      # both or neither
            mid = slp = None
        specs.append(FeatureSpec(
            name=name,
            family=str(r.get("family", "Other")),
            sign=str(r.get("sign", "unconstrained")).strip().lower(),
            adstock_decay=_f("adstock_decay"),
            coef_lower=_f("coef_lower"),
            coef_upper=_f("coef_upper"),
            role=str(r.get("role", "auto") or "auto").strip().lower(),
            # a missing/0/negative scale is meaningless — fall back to 1 rather
            # than dividing the data by zero or flipping its sign
            scale=(sc if sc and sc > 0 else 1.0),
            # saturation needs BOTH parameters to be meaningful; half a
            # specification is treated as "no saturation" rather than guessed
            sat_midpoint=(mid if mid and 0 < mid <= 1 else None),
            sat_slope=(slp if slp and slp > 0 else None),
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


def assemble_matrix(df: pd.DataFrame, specs: list,
                    sat_refs: Optional[dict] = None) -> pd.DataFrame:
    """Build the design matrix in the standard MMM order:

        raw  →  adstock (carry-over)  →  scale (units)  →  Hill (saturation)

    Adstock first because carry-over is about *when* the impression lands;
    saturation last because diminishing returns apply to the accumulated
    pressure, not to each week's raw spend. (Same order as Robyn/Meridian.)

    `sat_refs` maps variable -> the reference scale its Hill midpoint is a
    fraction of. Callers pass refs computed on TRAINING rows only; when a
    saturated variable has no ref supplied, this falls back to the max of the
    column it was handed — fine for a one-shot transform, wrong for a
    train/validate split, which is why run_slice always supplies them.
    """
    sat_refs = sat_refs or {}
    out = {}
    for s in specs:
        x = df[s.name].astype(float).values
        if s.adstock_decay is not None:
            x = adstock(x, s.adstock_decay)
        sc = getattr(s, "scale", 1.0) or 1.0
        if sc != 1.0:
            x = x / sc
        mid = getattr(s, "sat_midpoint", None)
        slp = getattr(s, "sat_slope", None)
        if mid is not None and slp is not None:
            # note: Hill is invariant to `scale` (the reference scales with
            # the data), so a saturated variable's coefficient is already in
            # volume units — the scale simply stops mattering for it
            x = hill(x, mid, slp, ref=sat_refs.get(s.name))
        out[s.name] = x
    return pd.DataFrame(out, index=df.index)


def saturation_refs_for(df: pd.DataFrame, specs: list,
                        rows: Optional[slice] = None) -> dict:
    """Reference scale per saturated variable, from `rows` (training only):
    the max of the adstocked, scaled series before saturation."""
    refs = {}
    for s in specs:
        if getattr(s, "sat_midpoint", None) is None \
                or getattr(s, "sat_slope", None) is None:
            continue
        x = df[s.name].astype(float).values
        if s.adstock_decay is not None:
            x = adstock(x, s.adstock_decay)
        sc = getattr(s, "scale", 1.0) or 1.0
        if sc != 1.0:
            x = x / sc
        refs[s.name] = saturation_ref(x[rows] if rows is not None else x)
    return refs


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
                     start_with: Optional[list] = None,
                     family_of: Optional[dict] = None,
                     max_per_family: Optional[dict] = None) -> list:
    """Forward selection by lowest p-value, keeping additions significant.
    `start_with` (force-include) variables are seeded into the model.

    `max_per_family` caps how many variables of a family may enter (Alex:
    "I want only two to three trade, but I want all media"). The cap is
    applied at ENTRY, not by trimming afterwards, so the family spends its
    budget on its most significant members: once Trade holds 3, further Trade
    candidates are simply not offered and the search continues with the rest —
    which is what "be more restrictive by group" has to mean if the remaining
    families are still to be selected properly.

    Forced variables are seeded regardless and may push a family over its cap:
    the client paid for them (same precedence as VIF pruning). They do count
    toward the budget, so a cap of 2 with 2 forced Trade variables admits no
    further Trade.
    """
    family_of = family_of or {}
    caps = {str(k): int(v) for k, v in (max_per_family or {}).items()
            if v is not None and str(v) != "" and int(v) > 0}
    selected = [c for c in (start_with or []) if c in X.columns]
    counts: dict = {}
    for c in selected:
        f = family_of.get(c)
        counts[f] = counts.get(f, 0) + 1
    remaining = [c for c in X.columns if c not in selected]
    yv = y.values.astype(float)
    ones = np.ones((len(X), 1))
    while remaining:
        base = X[selected].values if selected else np.empty((len(X), 0))
        best_p, best_c = 1.0, None
        for c in remaining:
            fam = family_of.get(c)
            if fam in caps and counts.get(fam, 0) >= caps[fam]:
                continue                    # this family is full
            Xd = np.column_stack([ones, base, X[c].values])
            p = _pvalue_of_last(Xd, yv)
            if p < best_p:
                best_p, best_c = p, c
        if best_c is not None and best_p < p_enter:
            selected.append(best_c)
            remaining.remove(best_c)
            fam = family_of.get(best_c)
            counts[fam] = counts.get(fam, 0) + 1
        else:
            break                           # nothing significant and allowed
    return selected


# ---------------------------------------------------------------------------
# 5. Constrained final fit
# ---------------------------------------------------------------------------

def _bounds_from_specs(specs_by_name: dict, cols: list):
    """(lower, upper) bounds per coefficient. Custom bounds in the config
    override the sign default; intercept is always unconstrained.

    Bounds are written by the analyst in ORIGINAL units ("this coefficient
    must stay under 1"), but the fit happens on x/scale, where the coefficient
    is `scale` times larger. So a bound must be multiplied by the scale to
    keep meaning the same thing — otherwise setting a scale would silently
    tighten every custom bound by a factor of 1,000. Sign constraints need no
    such treatment: dividing by a positive scale can't flip a sign."""
    lo, hi = [-np.inf], [np.inf]  # intercept
    for c in cols:
        s = specs_by_name[c]
        sc = getattr(s, "scale", 1.0) or 1.0
        if s.coef_lower is not None or s.coef_upper is not None:
            lo.append(s.coef_lower * sc if s.coef_lower is not None else -np.inf)
            hi.append(s.coef_upper * sc if s.coef_upper is not None else np.inf)
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
    # COMPARABLE coefficients (Arko 2026-07-27). A raw coefficient can't be
    # compared across variables because the inputs live on different supports
    # — "you can't really compare the coefficient for ACV weighted
    # distribution versus the five-year inflation expectations or gas price,
    # because the support you're using to build the model is not comparable".
    # These put every driver in the SAME unit — volume — by multiplying the
    # coefficient by how much the variable actually moves:
    #   beta_std   = β × SD(x)      volume per 1 standard-deviation move
    #   beta_range = β × (max−min)  volume across the full observed range
    # Both are invariant to `scale` (β shrinks exactly as SD grows), so they
    # are stable no matter how the analyst sets the units, and neither
    # changes the model: they are read-outs, not a re-parameterization.
    beta_std: pd.Series = field(default_factory=pd.Series)
    beta_range: pd.Series = field(default_factory=pd.Series)
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

    # comparable coefficients — same unit (volume) for every driver
    sd = X.std(ddof=0)
    rng = X.max() - X.min()
    beta_std = pd.Series({c: float(coef[c] * sd[c]) for c in cols})
    beta_range = pd.Series({c: float(coef[c] * rng[c]) for c in cols})

    return FitResult(cols, coef, fitted, resid, r2, adj_r2, mape,
                     contrib, tstats, vif, beta_std, beta_range,
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


# ─────────────────────────────────────────────────────────────────────────
# CONTRIBUTION vs DUE-TO  (Arko + Alex, 2026-07-27)
#
# These are two different numbers and the tool was showing only the first
# under the second's name:
#   CONTRIBUTION = how much volume this driver accounts for IN a period
#                  (a level: Σ βᵢ·xᵢ over the period's weeks).
#   DUE-TO       = how much of the CHANGE between two periods this driver
#                  explains (a delta: contribution in B − contribution in A).
# "How is the change in macro variables impacting my sales year to year" is a
# due-to question; the period totals alone cannot answer it. Both are kept —
# the due-to is what most readers of the deck actually care about.
# ─────────────────────────────────────────────────────────────────────────

def contributions_by_period(fit: FitResult, dates: pd.Series,
                            periods: dict) -> pd.DataFrame:
    """Total signed contribution per driver (rows) per named period (cols).
    `periods` maps a label -> a list/array of positional week indexes."""
    tbl = fit.contributions.reset_index(drop=True)
    out = {}
    for label, idx in periods.items():
        idx = [i for i in idx if 0 <= i < len(tbl)]
        out[label] = (tbl.iloc[idx].sum() if idx
                      else pd.Series(0.0, index=tbl.columns))
    return pd.DataFrame(out)


def due_to_change(totals: pd.DataFrame, period_a: str, period_b: str,
                  weeks: Optional[dict] = None,
                  normalize: bool = False) -> pd.DataFrame:
    """Period-over-period due-to: what the move from A to B is DUE TO.

    Columns: the two period levels, the absolute change (B − A), that change
    as a % of A, and each driver's share of the total modeled change.
    `normalize=True` compares per-week averages instead of totals — use it
    when the two periods have different lengths (e.g. latest 4 wks vs a year),
    otherwise a longer period looks bigger purely because it is longer.
    """
    a = totals[period_a].astype(float).copy()
    b = totals[period_b].astype(float).copy()
    if normalize and weeks:
        na, nb = max(int(weeks.get(period_a, 0)), 1), max(int(weeks.get(period_b, 0)), 1)
        a, b = a / na, b / nb
    change = b - a
    denom = a.abs().replace(0, np.nan)
    total_change = float(change.sum())
    out = pd.DataFrame({
        period_a: a, period_b: b,
        "due_to": change,
        "pct_change": (change / denom) * 100,
        "share_of_change_pct": (change / total_change * 100
                                if total_change else np.nan),
    })
    return out.sort_values("due_to", key=lambda s: s.abs(), ascending=False)


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
        df_all = df.copy()
        df_all["date"] = df_all["Time"].apply(_parse_week)
        df_full = df_all[df_all["Geography"] == channel].copy()
        df_full = df_full.sort_values("date").reset_index(drop=True)
    else:
        df_all = None
        df_full = load_slice(path, brand_sheet, channel)
    if cfg.target not in df_full.columns:
        raise ValueError(f"Target '{cfg.target}' not in data")

    # --- FIXED calendar window (Alex 2026-07-27) --------------------------
    # Anchor precedence: explicit cfg.window_end > dataset-wide latest week
    # (read from the workbook) > latest week across ALL channels of the sheet
    # we were handed. Never the single slice's own last row — that is exactly
    # the bug: a delisted slice would silently model 2023 data and emit
    # due-tos for a period nobody is looking at.
    anchor = cfg.window_end
    if anchor is None and path:
        try:
            anchor = dataset_anchor_week(path)
        except Exception:
            anchor = None
    if anchor is None and df_all is not None:
        anchor = df_all["date"].max()
    if anchor is None:
        anchor = df_full["date"].max()
    win_start, win_end = resolve_window(anchor, cfg.model_weeks)

    in_win = (df_full["date"] >= win_start) & (df_full["date"] <= win_end)
    df_win = df_full[in_win].sort_values("date").reset_index(drop=True)
    n_win = len(df_win)
    tgt_vals = pd.to_numeric(df_win[cfg.target], errors="coerce").fillna(0) \
        if n_win else pd.Series(dtype=float)
    n_nonzero = int((tgt_vals != 0).sum())
    if n_win < cfg.min_weeks or n_nonzero < cfg.min_nonzero_weeks:
        raise InsufficientWindowData(
            f"only {n_win} weeks ({n_nonzero} with non-zero {cfg.target}) "
            f"inside the modeling window "
            f"{win_start:%Y-%m-%d}..{win_end:%Y-%m-%d} — needs "
            f"{cfg.min_weeks} weeks / {cfg.min_nonzero_weeks} selling weeks; "
            "not modeled",
            weeks=n_win, nonzero=n_nonzero,
            window_start=win_start, window_end=win_end)
    df_full = df_win

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

    # Hill reference scales from TRAINING weeks only (everything before the
    # reserved validation tail). If the reference came from the whole window,
    # the saturation transform itself would have seen the holdout — a subtle
    # leak that would flatter the out-of-sample number.
    _train_rows = slice(0, max(len(df_full) - val_hold, 1))
    sat_refs = saturation_refs_for(df_full, specs, rows=_train_rows)

    def _assemble(a, b):
        dfx = df_full.iloc[a:b].reset_index(drop=True)
        return (dfx, assemble_matrix(dfx, specs, sat_refs=sat_refs),
                dfx[spec_names].copy(), dfx[cfg.target].astype(float))

    def _select(Xtr, ytr):
        """VIF prune + forward stepwise + dead-coef prune -> selected names."""
        Xz = (Xtr - Xtr.mean()) / Xtr.std(ddof=0)
        Xz = Xz.loc[:, Xz.std() > 0]
        favail = [c for c in force if c in Xz.columns]
        vif_keep = prune_by_vif(Xz, threshold=cfg.vif_threshold, protect=favail)
        fam_of = {n: s.family for n, s in specs_by_name.items()}
        selected = forward_stepwise(
            Xz[vif_keep], ytr, p_enter=cfg.p_enter, start_with=favail,
            family_of=fam_of, max_per_family=cfg.max_per_family)
        if not selected:
            # nothing cleared p_enter — fall back to a few candidates so the
            # slice still produces a model, but honour the caps here too,
            # otherwise the fallback would quietly violate the limits the
            # analyst just set.
            selected, seen = [], {}
            for c in vif_keep:
                f = fam_of.get(c)
                cap = (cfg.max_per_family or {}).get(f)
                if cap and seen.get(f, 0) >= int(cap):
                    continue
                selected.append(c)
                seen[f] = seen.get(f, 0) + 1
                if len(selected) >= 5:
                    break
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

    # REPORTED window = the fixed calendar window, in full. df_full was already
    # clipped to [win_start, win_end], so this is every week in the window and
    # NOTHING outside it — no stale pre-window history can reach the due-tos.
    df, X_all, X_raw, y = _assemble(0, n)

    # VALIDATION: SELECT on training (no tail leakage) + score the reserved tail.
    # Both fits now live inside the SAME fixed window (the tail is carved out of
    # it rather than the window sliding back), so the validation model never
    # reaches for data outside the period we said we were modeling.
    holdout_mape = np.nan
    pred_te = yte = np.array([])
    if val_hold > 0:
        sv = n - val_hold
        selected = _select(X_all.iloc[:sv], y.iloc[:sv])
        vfit = constrained_fit(X_all[selected].iloc[:sv], y.iloc[:sv],
                               specs_by_name)
        Xte = np.column_stack([np.ones(val_hold),
                               X_all[selected].iloc[sv:].values])
        pred_te = Xte @ vfit.coef.values
        yte = y.iloc[sv:].values
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
        # the fixed window this model was built on, for captions/exports
        "window_start": win_start, "window_end": win_end,
        "n_weeks_window": n_win, "n_weeks_selling": n_nonzero,
        "sat_refs": sat_refs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. MEDIA CURVE OPTIMIZATION  (Alex 2026-07-27, F2 + F3)
#
#   "The ad stock that you put in — 0.5. Does it always have to be 0.5? Can we
#    create an optimization engine that will pick the best ad stock based on
#    correlation with the data, predictive power in a regression? We can do the
#    same thing with saturation. Is there a certain slope and midpoint that
#    best fits the data? … If we want a fully automated system, that would be
#    dope as hell."
#
# Search space per media variable: decay d (carry-over) × midpoint γ × slope s
# (Hill saturation), including the option of NO saturation.
#
# ── The part that matters most: what it is scored on ──────────────────────
# Tuning curve shapes is a search over a large space, so whatever set you
# score on gets over-fitted — that is not a risk, it is a certainty. So the
# window is split THREE ways, not two:
#
#     [ ─────────── train ─────────── | inner validation | OUTER HOLDOUT ]
#                                        13 wks             13 wks
#
# The optimizer fits on `train` and scores candidates on `inner validation`.
# The OUTER holdout is never touched during the search, so the holdout MAPE
# the dashboard reports afterwards is still an honest out-of-sample number and
# is directly comparable with a model that was never optimized. If the
# optimizer had been scored on the reported holdout, every optimized model
# would look better and none of them would be.
#
# Search strategy: coordinate descent — one variable at a time, others held at
# their current values, a couple of passes. Full joint search over k media
# variables is (grid)^k; coordinate descent is k×(grid) per pass and in
# practice finds the same optimum for a near-separable problem like this.
# ═══════════════════════════════════════════════════════════════════════════

# Defaults chosen to span the behaviour Alex described without exploding the
# search: decay 0 (no carry-over) → 0.8 (long TV-style tail); midpoint from
# early saturation (20% of peak) to late (80%); slope <1 concave, >1 the
# S-curve. `None` in the saturation grid = leave the variable linear, so the
# optimizer can always decline to saturate.
DECAY_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)
MIDPOINT_GRID = (0.2, 0.35, 0.5, 0.65, 0.8)
SLOPE_GRID = (0.5, 1.0, 1.5, 2.5)


def _fold_mape(X: pd.DataFrame, y: pd.Series, cols: list, specs_by_name: dict,
               n_train: int, n_score: int) -> float:
    """Fit on rows [0, n_train), score the next n_score rows."""
    end = n_train + n_score
    if not cols or end > len(X) or n_train < len(cols) + 2:
        return np.inf
    try:
        f = constrained_fit(X[cols].iloc[:n_train], y.iloc[:n_train],
                            specs_by_name)
    except Exception:
        return np.inf
    Xte = np.column_stack([np.ones(n_score),
                           X[cols].iloc[n_train:end].values])
    pred = Xte @ f.coef.values
    yte = y.iloc[n_train:end].values
    nz = yte != 0
    if not nz.any():
        return np.inf
    err = float(np.mean(np.abs((yte[nz] - pred[nz]) / yte[nz]))) * 100
    return err if np.isfinite(err) else np.inf


def _score_structure(X: pd.DataFrame, y: pd.Series, cols: list,
                     specs_by_name: dict, folds: list) -> float:
    """Mean MAPE across ROLLING-ORIGIN folds.

    A single inner validation window turned out to be too easy to over-fit:
    on Brand 1 × Channel 1 the search improved a lone 13-week inner window
    from 11.2% to 9.7% while making the untouched outer holdout WORSE (19.6%
    → 21.9%). It had tuned the curves to one particular quarter.

    So candidates are scored on several expanding-window folds instead —
    train on everything before a fold, score the fold, average. A curve shape
    now has to help in several different periods to be adopted, which is what
    "this decay is real" actually means. Structure is held fixed throughout so
    the score reflects the curve shapes and nothing else.
    """
    scores = [_fold_mape(X, y, cols, specs_by_name, n_tr, n_sc)
              for n_tr, n_sc in folds]
    ok = [s for s in scores if np.isfinite(s)]
    if not ok:
        return np.inf
    return float(np.mean(ok))


def _fold_scores(X: pd.DataFrame, y: pd.Series, cols: list,
                 specs_by_name: dict, folds: list) -> list:
    return [_fold_mape(X, y, cols, specs_by_name, n_tr, n_sc)
            for n_tr, n_sc in folds]


def optimize_media_curves(df_window: pd.DataFrame, specs: list, target: str,
                          selected: Optional[list] = None,
                          inner_holdout: int = 13, outer_holdout: int = 13,
                          n_folds: int = 3, passes: int = 2,
                          decay_grid=DECAY_GRID, midpoint_grid=MIDPOINT_GRID,
                          slope_grid=SLOPE_GRID,
                          min_improvement: float = 0.5,
                          progress=None) -> dict:
    """Search adstock decay + Hill saturation per media variable.

    Returns {"params": {var: {adstock_decay, sat_midpoint, sat_slope}},
             "baseline_inner_mape", "optimized_inner_mape", "n_fits",
             "per_variable": [...]}  — parameters only. The caller writes them
    to the config and re-runs normally, so nothing here can quietly become a
    second, divergent modeling path.

    A variable's new curve is adopted only if it improves the inner-validation
    MAPE by at least `min_improvement` percentage points; otherwise it keeps
    what it had. Without that floor the search would happily trade a 0.01%
    "gain" for an implausible curve shape.
    """
    n = len(df_window)
    n_outer = min(outer_holdout, max(n // 6, 0))
    n_avail = n - n_outer                     # optimizer's whole universe
    n_inner = min(inner_holdout, max(n_avail // 6, 0))
    # Rolling-origin folds inside the optimizer's universe: the LAST fold ends
    # at n_avail, each earlier fold shifts back by one fold length.
    folds = []
    for i in range(max(n_folds, 1)):
        end = n_avail - i * n_inner
        n_tr = end - n_inner
        if n_tr >= 15 and n_inner >= 4:
            folds.append((n_tr, n_inner))
    folds.reverse()
    n_train = folds[-1][0] if folds else 0
    if not folds:
        return {"params": {}, "baseline_inner_mape": np.nan,
                "optimized_inner_mape": np.nan, "n_fits": 0,
                "per_variable": [],
                "note": (f"window too short to optimize safely "
                         f"({n} weeks: needs ≥15 training weeks and a ≥4-week "
                         f"validation fold after reserving the outer holdout)")}

    # everything the optimizer sees excludes the OUTER holdout entirely
    df_opt = df_window.iloc[:n_avail].reset_index(drop=True)
    y = df_opt[target].astype(float)

    by_name = {s.name: s for s in specs}
    media = [s.name for s in specs if s.adstock_decay is not None]
    if selected:                      # optimize what is actually in the model
        media = [m for m in media if m in selected]
    if not media:
        return {"params": {}, "baseline_inner_mape": np.nan,
                "optimized_inner_mape": np.nan, "n_fits": 0,
                "per_variable": [], "note": "no media variables in the model"}

    cols = list(selected) if selected else [s.name for s in specs]
    cols = [c for c in cols if c in by_name]

    # current parameter state, seeded from the config
    state = {m: {"adstock_decay": by_name[m].adstock_decay,
                 "sat_midpoint": by_name[m].sat_midpoint,
                 "sat_slope": by_name[m].sat_slope} for m in media}

    def build(state_):
        """Design matrix under a candidate parameter state, with Hill
        references taken from TRAINING rows only."""
        trial = []
        for s in specs:
            if s.name in state_:
                st = state_[s.name]
                trial.append(FeatureSpec(
                    name=s.name, family=s.family, sign=s.sign,
                    adstock_decay=st["adstock_decay"],
                    coef_lower=s.coef_lower, coef_upper=s.coef_upper,
                    role=s.role, scale=s.scale,
                    sat_midpoint=st["sat_midpoint"],
                    sat_slope=st["sat_slope"]))
            else:
                trial.append(s)
        # references from the FIRST fold's training rows — the earliest and
        # therefore most conservative slice of history any fold trains on
        refs = saturation_refs_for(df_opt, trial, rows=slice(0, folds[0][0]))
        return (assemble_matrix(df_opt, trial, sat_refs=refs),
                {t.name: t for t in trial})

    X0, sbn0 = build(state)
    base_folds = _fold_scores(X0, y, cols, sbn0, folds)
    base_score = float(np.mean([v for v in base_folds if np.isfinite(v)])
                       or np.inf)
    best_score, best_folds, n_fits = base_score, base_folds, 1

    candidates = [(d, None, None) for d in decay_grid] + \
                 [(d, m, sl) for d in decay_grid
                  for m in midpoint_grid for sl in slope_grid]

    for p in range(passes):
        improved_this_pass = False
        for var in media:
            keep = dict(state[var])
            local_best, local_score, local_folds = keep, best_score, best_folds
            for (d, m, sl) in candidates:
                trial_state = dict(state)
                trial_state[var] = {"adstock_decay": d, "sat_midpoint": m,
                                    "sat_slope": sl}
                Xc, sbnc = build(trial_state)
                fs = _fold_scores(Xc, y, cols, sbnc, folds)
                n_fits += 1
                finite = [v for v in fs if np.isfinite(v)]
                if not finite:
                    continue
                sc = float(np.mean(finite))
                # STRICT DOMINANCE: a curve must not be worse in ANY fold.
                # Averaging alone let a shape that happened to suit one
                # quarter win while degrading another — precisely the
                # over-fit this whole three-way split exists to prevent.
                if any(f > b_ + 1e-9 for f, b_ in zip(fs, best_folds)
                       if np.isfinite(f) and np.isfinite(b_)):
                    continue
                if sc < local_score - 1e-12:
                    local_best, local_score, local_folds = \
                        trial_state[var], sc, fs
            # adopt only a materially better curve
            if local_score < best_score - min_improvement:
                state[var] = local_best
                best_score, best_folds = local_score, local_folds
                improved_this_pass = True
            if progress:
                progress(f"pass {p+1}: {var} → "
                         f"decay {state[var]['adstock_decay']}, "
                         f"sat {state[var]['sat_midpoint']}/"
                         f"{state[var]['sat_slope']} "
                         f"(inner MAPE {best_score:.2f}%)")
        if not improved_this_pass:
            break                     # converged — further passes can't help

    per_var = [{"variable": v,
                "adstock_decay": state[v]["adstock_decay"],
                "sat_midpoint": state[v]["sat_midpoint"],
                "sat_slope": state[v]["sat_slope"],
                "changed": state[v] != {"adstock_decay": by_name[v].adstock_decay,
                                        "sat_midpoint": by_name[v].sat_midpoint,
                                        "sat_slope": by_name[v].sat_slope}}
               for v in media]
    return {"params": state, "baseline_inner_mape": base_score,
            "optimized_inner_mape": best_score,
            "baseline_fold_mapes": [float(v) for v in base_folds],
            "optimized_fold_mapes": [float(v) for v in best_folds],
            "n_fits": n_fits,
            "per_variable": per_var, "n_train": n_train, "n_inner": n_inner,
            "n_outer": n_outer, "folds": folds}


def response_curve(spec: FeatureSpec, coef: float, ref: float,
                   x_max: Optional[float] = None, points: int = 60) -> dict:
    """Points for drawing one media variable's response curve: modeled volume
    against (adstocked, scaled) execution. Linear variables give a straight
    line; saturated ones give the Hill curve, where the flattening IS the
    diminishing return."""
    hi = float(x_max if x_max is not None else (ref or 1.0)) * 1.25 or 1.0
    xs = np.linspace(0.0, hi, points)
    if spec.sat_midpoint is not None and spec.sat_slope is not None:
        ys = coef * hill(xs, spec.sat_midpoint, spec.sat_slope, ref=ref)
        half = spec.sat_midpoint * (ref or 0.0)
    else:
        ys = coef * xs
        half = None
    return {"x": [float(v) for v in xs], "y": [float(v) for v in ys],
            "half_saturation": (None if half is None else float(half)),
            "saturated": half is not None}


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

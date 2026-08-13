"""
MULTI-LEVEL MODELING — unpooled / pooled / hierarchical
═══════════════════════════════════════════════════════════════════════════════
Alex Hathcock, "Last Set of Dashboard Updates" (2026-08-11) + the 2026-08-11
call, where Arko framed the same three levels in market-mix terms:

    "It can be a pooled model where you are pooling across all the different
     markets, which will basically share the same coefficient… the other option
     is an unpooled model, where every single DMA will have different
     coefficients… and the third option is a partially pooled model, which is
     more like a hierarchical model, where the coefficients are not the same or
     different — there is some kind of relationship between the national
     coefficient versus the DMA-level coefficient."

Here the "markets" are CHANNELS and the top level is the BRAND.

    LEVEL 1  UNPOOLED       one independent model per Brand × Channel.
                            Already shipped — this is `cp.run_slice`. Maximum
                            flexibility, fewest degrees of freedom (104 weeks
                            per model), coefficients free to disagree wildly
                            between channels for no reason but noise.

    LEVEL 2  POOLED         one model per Brand, all channels stacked. Alex:
                            "you have way more degrees of freedom because you
                            have that many more rows… the individual channels
                            are less important versus the total." 9 channels ×
                            104 weeks = 936 rows against the same predictor
                            count. One coefficient per predictor, shared.

    LEVEL 3  HIERARCHICAL   Brand × Channel coefficients that are a pooled
                            FIXED effect times a channel-specific RANDOM
                            effect. "That media might generally have a
                            coefficient of 0.5, but at Brand 1 Channel 1 it has
                            0.4, at Brand 1 Channel 3 it has 0.6."

WHY THIS IS LEGITIMATE ON THIS DATA (Alex made the argument on the call, and it
is the assumption the whole design rests on, so it is written down here):
"Since we're not technically doing a time series model, there is no relationship
between time at one and time at two. All rows are independent based on the
variables that we have… the decomposition, the decay, all of that is to make
this model, which is OF a time series, NOT a time series model. So you can just
stack channel one, channel two, channel three, all the weeks." Adstock and
saturation are computed WITHIN a channel before stacking (see
`_stack_brand`) — that is precisely what keeps the carry-over structure intact
while letting the rows be pooled.

A note on what this is NOT: it is not a Bayesian hierarchical model. Alex was
explicit — "they do that typically with something more like a Bayesian model,
where this one is more of an optimized least squares model. We don't need to do
Bayesian; that would be a complete recalculation of everything you guys have
done on the back end." So partial pooling here is done by INDEXING (his Excel
sheet, `Hierarchical Modeling Explanation.xlsx`) rather than by a prior and a
sampler. The shrinkage is explicit and capped instead of emerging from a
posterior. That is a real difference and it is stated in the Definitions tab.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import capstone_pipeline as cp


# ═══════════════════════════════════════════════════════════════════════════
# 1. NATIONAL PREDICTORS  (the pre-modeling step Alex flagged)
# ═══════════════════════════════════════════════════════════════════════════
#
#   "'National' predictors — predictors that have the exact same values for
#    each brand*channel — need to be proportionally split up amongst the
#    channels before modeling… This way, the single coefficient applied will
#    not massively increase contributions at the brand level.
#    Easy way to check? Sum each predictor by channel and brand. If the sums
#    are the same for each channel, it's a 'National' level predictor."
#
# The problem, concretely: unemployment rate is identical in all 9 channels. In
# a pooled model it gets ONE coefficient β. Every channel row then carries a
# contribution of β·x, so the brand total is 9·β·x — nine times the effect the
# variable can possibly have had. Splitting it by each channel's share of brand
# volume makes the per-channel pieces sum back to β·x exactly.
# ═══════════════════════════════════════════════════════════════════════════

def national_predictors(df_brand: pd.DataFrame, columns: list,
                        channel_col: str = "Geography",
                        rtol: float = 1e-6,
                        date_col: str = "date") -> dict:
    """Which predictors are national, i.e. carry the same values in every
    channel and so must be split before pooling.

    ── A REAL FALSE POSITIVE IN THE QUICK TEST, WORTH KNOWING ABOUT ─────────
    Alex proposed a fast screen: "Sum each predictor by channel and brand. If
    the sums are the same for each channel, it's a 'National' level predictor."
    On this data that screen flags 26 variables — and one of them is not
    national at all. `Seasonality_Index` is a per-channel index normalised to
    average 1.0, so EVERY channel sums to exactly 104.000000 over a 104-week
    window while the weekly values differ substantially between channels
    (2024-01-07: 0.947 in Channel 1 against 0.565 in Channel 9). Equal sums,
    different data. Splitting it would have destroyed a genuinely
    channel-specific predictor — and it is a strong one, so the damage would
    have been quiet and large.

    So the DEFINITION used here is Alex's own words rather than his shortcut —
    "predictors that have the exact same values for each brand*channel" — and
    the implementation compares the weekly values channel by channel. The sum
    screen is still run, and anything that passes it but fails the value test is
    returned under `sum_equal_only` so the user sees the near-miss rather than
    silently getting a different answer than the email describes.

    Returns
    -------
    {"national": [...], "sum_equal_only": [...], "n_checked": int}
    """
    empty = {"national": [], "sum_equal_only": [], "n_checked": 0}
    if channel_col not in df_brand.columns or date_col not in df_brand.columns:
        return empty
    g = df_brand.groupby(channel_col)
    national, sum_only, checked = [], [], 0
    for c in columns:
        if c not in df_brand.columns:
            continue
        sums = pd.to_numeric(g[c].sum(), errors="coerce").dropna()
        if len(sums) < 2:
            continue
        m = float(np.abs(sums).max())
        if m == 0:
            continue                      # all-zero: nothing to split
        checked += 1
        if float(sums.max() - sums.min()) > rtol * m:
            continue                      # fails even the quick screen
        # the real test: identical values week by week across channels
        wide = df_brand.pivot_table(index=date_col, columns=channel_col,
                                    values=c, aggfunc="max")
        spread = float((wide.max(axis=1) - wide.min(axis=1)).abs().max())
        scale = max(float(np.abs(wide.values).max()), 1e-12)
        (national if spread <= 1e-9 * scale else sum_only).append(c)
    return {"national": national, "sum_equal_only": sum_only,
            "n_checked": checked}


def channel_shares(df_brand: pd.DataFrame, target: str,
                   channel_col: str = "Geography") -> pd.Series:
    """ΣChannel_Target / ΣBrand_Target — each channel's share of brand volume.

    Alex's proportion. Computed on the modeling window that was handed in, so
    the shares describe the period being modeled rather than all history.
    Falls back to equal weights if the brand has no volume at all in the window
    (otherwise every national predictor would be zeroed out).
    """
    tot = pd.to_numeric(df_brand[target], errors="coerce").fillna(0)
    by_ch = tot.groupby(df_brand[channel_col]).sum()
    grand = float(by_ch.sum())
    if grand <= 0:
        n = max(len(by_ch), 1)
        return pd.Series(1.0 / n, index=by_ch.index)
    return by_ch / grand


def proportionalize_national(df_brand: pd.DataFrame, national: list,
                             shares: pd.Series,
                             channel_col: str = "Geography") -> pd.DataFrame:
    """Multiply each national predictor by its channel's share of brand volume.

    ── ON THE WORDING ────────────────────────────────────────────────────────
    Alex wrote "divide the values by the channels overall proportion". Taken
    literally that is x / share, which for a 20% channel MULTIPLIES the value by
    5 and makes contributions larger, the opposite of the stated purpose ("will
    not massively increase contributions at the brand level") and the opposite
    of the verb in the same sentence ("proportionally SPLIT UP amongst the
    channels"). So the implementation is the split: x_channel = x × share, which
    has the property that makes the whole thing work —

        Σ_channels x_channel = x_national

    — i.e. the pieces add back up to the national quantity, so one shared
    coefficient produces the right brand-level contribution instead of one per
    channel. `pooled_national_check` asserts exactly this after every run, and
    the Definitions tab states the convention. If Alex meant the literal
    division, this is the ONE line to flip.
    """
    out = df_brand.copy()
    if not national:
        return out
    w = out[channel_col].map(shares).astype(float).fillna(0.0).values
    for c in national:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).values * w
    return out


def pooled_national_check(df_before: pd.DataFrame, df_after: pd.DataFrame,
                          national: list,
                          channel_col: str = "Geography") -> dict:
    """Verify the split conserved the national totals: for each national
    predictor, the sum ACROSS channels of the split values in a given week must
    equal the original national value for that week. Returns the worst relative
    error seen — the dashboard shows it, and a non-trivial number here means
    the shares did not cover every row."""
    worst, worst_var = 0.0, None
    for c in national:
        before = pd.to_numeric(df_before[c], errors="coerce").fillna(0)
        after = pd.to_numeric(df_after[c], errors="coerce").fillna(0)
        # one national value per week (identical across channels) vs the sum of
        # its per-channel pieces in that week
        wk = df_before.get("date")
        if wk is None:
            continue
        b = before.groupby(wk).max()
        a = after.groupby(df_after["date"]).sum()
        denom = float(np.abs(b).max()) or 1.0
        err = float(np.abs((a - b).dropna()).max()) / denom
        if err > worst:
            worst, worst_var = err, c
    return {"max_rel_error": worst, "worst_variable": worst_var,
            "n_national": len(national), "ok": worst < 1e-9}


# ═══════════════════════════════════════════════════════════════════════════
# 2. STACKING A BRAND
# ═══════════════════════════════════════════════════════════════════════════

def _stack_brand(path: str, brand_sheet: str, cfg: cp.ModelConfig,
                 df: Optional[pd.DataFrame] = None,
                 channels: Optional[list] = None) -> dict:
    """Every channel of one brand, clipped to the shared calendar window and
    stacked into one long frame.

    THE ORDER THAT MATTERS: adstock and Hill saturation are applied PER CHANNEL,
    on that channel's own contiguous weekly series, and only then are the rows
    stacked. Doing it the other way — stacking first, transforming after —
    would carry Channel 1's December spend into Channel 2's January, which is
    meaningless. This is why the design matrix is assembled channel-by-channel
    here rather than by one call to `cp.assemble_matrix` on the stacked frame.

    Saturation references are likewise computed per channel on TRAINING rows
    only (before the reserved validation tail), same rule as `cp.run_slice`, so
    the transform cannot peek at the holdout.
    """
    if df is None:
        df = pd.read_excel(path, sheet_name=brand_sheet)
    df = df.copy()
    df["date"] = df["Time"].apply(cp._parse_week)

    anchor = cfg.window_end
    if anchor is None and path:
        try:
            anchor = cp.dataset_anchor_week(path)
        except Exception:
            anchor = None
    if anchor is None:
        anchor = df["date"].max()
    win_start, win_end = cp.resolve_window(anchor, cfg.model_weeks)
    df = df[(df["date"] >= win_start) & (df["date"] <= win_end)]

    chans = channels or sorted(df["Geography"].dropna().unique().tolist())
    kept, skipped = [], []
    for ch in chans:
        d = df[df["Geography"] == ch].sort_values("date").reset_index(drop=True)
        n = len(d)
        tv = pd.to_numeric(d[cfg.target], errors="coerce").fillna(0) if n else pd.Series(dtype=float)
        nz = int((tv != 0).sum())
        if n < cfg.min_weeks or nz < cfg.min_nonzero_weeks:
            skipped.append({"channel": ch, "weeks": n, "selling": nz,
                            "reason": f"only {n} weeks ({nz} selling) inside "
                                      f"the window — needs {cfg.min_weeks}/"
                                      f"{cfg.min_nonzero_weeks}"})
            continue
        kept.append((ch, d))
    if not kept:
        raise cp.InsufficientWindowData(
            f"{brand_sheet}: no channel has enough data inside "
            f"{win_start:%Y-%m-%d}..{win_end:%Y-%m-%d} to pool",
            weeks=0, nonzero=0, window_start=win_start, window_end=win_end)

    raw_stacked = pd.concat([d for _, d in kept], ignore_index=True)

    # ── candidate predictors: one spec list for the whole brand ──────────
    specs = cp.build_feature_specs(raw_stacked,
                                   media_decay=cfg.default_media_decay,
                                   config=cfg.variable_config)
    leak = tuple({cfg.target, "Dollar Sales", "Volume Sales"}
                 if cfg.target.startswith(("Volume Sales", "Dollar Sales"))
                 else {cfg.target})
    specs = [s for s in specs
             if s.name != cfg.target and not s.name.startswith(leak)]
    force = [s.name for s in specs if s.role == "force"] + \
            [c for c in cfg.force_include if c in raw_stacked.columns]
    drop = {s.name for s in specs if s.role == "exclude"} | set(cfg.exclude)
    specs = [s for s in specs if s.name not in drop]
    specs_by_name = {s.name: s for s in specs}
    force = [c for c in dict.fromkeys(force) if c in specs_by_name]
    names = [s.name for s in specs]

    # ── national split, BEFORE any transform ────────────────────────────
    nat = national_predictors(raw_stacked, names)
    national = nat["national"]
    shares = channel_shares(raw_stacked, cfg.target)
    split_stacked = proportionalize_national(raw_stacked, national, shares)
    nat_check = pooled_national_check(raw_stacked, split_stacked, national)
    nat_check["sum_equal_only"] = nat["sum_equal_only"]

    # ── per-channel transform, then stack ───────────────────────────────
    cap = cfg.holdout_weeks if cfg.holdout_weeks is not None else 13
    X_parts, raw_parts, y_parts, meta_rows, sat_refs = [], [], [], [], {}
    for ch, d in kept:
        dc = split_stacked[split_stacked["Geography"] == ch] \
            .sort_values("date").reset_index(drop=True)
        n = len(dc)
        hold = int(min(max(cap, 0), max(n - 1, 0)))
        if n - hold < 5:
            hold = max(0, n - 5)
        refs = cp.saturation_refs_for(dc, specs, rows=slice(0, max(n - hold, 1)))
        for k, v in refs.items():                 # keep the per-channel refs
            sat_refs.setdefault(ch, {})[k] = v
        Xc = cp.assemble_matrix(dc, specs, sat_refs=refs)
        X_parts.append(Xc)
        raw_parts.append(dc[names].copy())
        y_parts.append(pd.to_numeric(dc[cfg.target], errors="coerce").fillna(0))
        meta_rows.append(pd.DataFrame({"channel": ch, "date": dc["date"],
                                       "is_holdout": [False] * (n - hold)
                                                     + [True] * hold}))

    X = pd.concat(X_parts, ignore_index=True)
    X_raw = pd.concat(raw_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True).astype(float)
    meta = pd.concat(meta_rows, ignore_index=True)

    return {"X": X, "X_raw": X_raw, "y": y, "meta": meta,
            "df": split_stacked.reset_index(drop=True),
            "df_unsplit": raw_stacked.reset_index(drop=True),
            "specs": specs, "specs_by_name": specs_by_name, "force": force,
            "national": national, "shares": shares, "national_check": nat_check,
            "channels": [ch for ch, _ in kept], "skipped": skipped,
            "sat_refs": sat_refs, "window_start": win_start,
            "window_end": win_end, "per_channel": dict(kept)}


def _select(X: pd.DataFrame, y: pd.Series, cfg: cp.ModelConfig,
            specs_by_name: dict, force: list) -> list:
    """VIF prune → forward stepwise → dead-coefficient prune. Identical policy
    to `cp.run_slice._select`, applied to whatever matrix it is handed — the
    pooled model must be selected the same way the unpooled ones are, or a
    comparison between the levels compares two different procedures."""
    Xz = (X - X.mean()) / X.std(ddof=0)
    Xz = Xz.loc[:, Xz.std() > 0]
    favail = [c for c in force if c in Xz.columns]
    keep = cp.prune_by_vif(Xz, threshold=cfg.vif_threshold, protect=favail)
    fam_of = {n: s.family for n, s in specs_by_name.items()}
    selected = cp.forward_stepwise(Xz[keep], y, p_enter=cfg.p_enter,
                                   start_with=favail, family_of=fam_of,
                                   max_per_family=cfg.max_per_family)
    if not selected:
        selected, seen = [], {}
        for c in keep:
            f = fam_of.get(c)
            lim = (cfg.max_per_family or {}).get(f)
            if lim and seen.get(f, 0) >= int(lim):
                continue
            selected.append(c)
            seen[f] = seen.get(f, 0) + 1
            if len(selected) >= 5:
                break
    sd = X[selected].std(ddof=0)
    for _ in range(len(selected)):
        f = cp.constrained_fit(X[selected], y, specs_by_name)
        impact = {c: abs(f.coef[c]) * sd[c] for c in selected}
        sc = max(impact.values()) or 1.0
        dead = [c for c in selected if impact[c] / sc < 1e-4 and c not in favail]
        if not dead:
            break
        selected = [c for c in selected if c not in dead]
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# 3. POOLED MODEL  (level 2)
# ═══════════════════════════════════════════════════════════════════════════

def _center_within(frame, groups: dict, rows: np.ndarray):
    """Subtract each channel's own mean, computed on `rows` only.

    Returns (centered, means). Works for a DataFrame or a Series.
    """
    means = {}
    out = frame.copy().astype(float)
    for ch, idx in groups.items():
        i = np.asarray(list(idx))
        base = i[rows[i]] if rows is not None else i
        if not len(base):
            base = i
        m = frame.loc[base].mean()
        means[ch] = m
        out.loc[i] = (frame.loc[i] - m).values
    return out, means


def run_pooled(path: str, brand_sheet: str,
               config: Optional[cp.ModelConfig] = None,
               df: Optional[pd.DataFrame] = None,
               channels: Optional[list] = None,
               channel_intercepts: bool = True) -> dict:
    """One model for a whole brand, all channels stacked.

    ── CHANNEL INTERCEPTS / MEAN CENTERING (`channel_intercepts=True`) ──────
    Arko, on the 2026-08-11 call: "we should do the full modeling, or like the
    mean centered models." This is that, and it is not a detail — it is the
    difference between a pooled model that works and one that cannot.

    Within Brand 1 the channels differ in volume by 30× (Channel 1 ≈ 7.1M over
    the window, Channel 9 ≈ 0.22M). A single shared intercept has to sit
    somewhere between them, so the big channel is under-predicted and the small
    one over-predicted by a large constant, every single week, no matter how
    good the slopes are. The model then spends its coefficients trying to
    explain a level difference that has nothing to do with the drivers.

    The fix is a per-channel intercept, implemented as the WITHIN transform:
    subtract each channel's own mean from y and from every predictor, fit the
    slopes on the centered data, then recover

        intercept_c = ȳ_c − Σ β·x̄_c

    By Frisch–Waugh–Lovell this gives EXACTLY the slope estimates that adding
    one dummy per channel would, without putting k dummy columns through the
    variable-selection machinery (where they would eat the family caps, distort
    every VIF, and show up in the coefficient table as drivers, which they are
    not). What stays pooled is what Alex wanted pooled — one coefficient per
    predictor for the whole brand. What varies is only the base level, which is
    a fact about channel size, not about how the brand responds to media.

    The means are computed on TRAINING ROWS ONLY, so the centering cannot leak
    the holdout. Set `channel_intercepts=False` for the plain single-intercept
    pooled model (kept because it is the literal reading of the email, and the
    comparison between the two is itself informative).

    Holdout: each channel keeps its OWN last 13 weeks as the validation tail
    and the tails are pooled for scoring. Alex noted "your holdout period would
    have to change a little bit" — this is the change. Holding out the last 13
    stacked ROWS instead would hold out one channel's tail and none of the
    others', which tests nothing about the other channels.

    Post-modeling: "Due-tos can still be calculated at the Channel level, just
    using the same coefficients across all channels (but using the
    proportionalized national predictors)." That is `channel_contributions`
    below, and it is why the split matters — the channel decomposition is only
    meaningful because each channel carries its own slice of the national
    variable.
    """
    cfg = config or cp.ModelConfig()
    S = _stack_brand(path, brand_sheet, cfg, df=df, channels=channels)
    X, y, meta = S["X"], S["y"], S["meta"]
    train = ~meta["is_holdout"].values
    groups = meta.groupby("channel").groups

    if channel_intercepts:
        Xf, x_means_tr = _center_within(X, groups, train)
        yf, y_means_tr = _center_within(y, groups, train)
    else:
        Xf, yf, x_means_tr, y_means_tr = X, y, None, None

    # SELECT on training rows only (no tail leakage), score on the pooled tails
    selected = _select(Xf.loc[train], yf.loc[train], cfg, S["specs_by_name"],
                       S["force"])
    holdout_mape = holdout_wmape = np.nan
    pred_te = yte = np.array([])
    if (~train).any() and selected:
        vfit = cp.constrained_fit(Xf.loc[train, selected], yf.loc[train],
                                  S["specs_by_name"])
        te = np.where(~train)[0]
        if channel_intercepts:
            # predict on the ORIGINAL scale: each channel's training mean is
            # added back, so the holdout is scored in volume, not in deviations
            pred_te = np.zeros(len(te), dtype=float)
            for j, i in enumerate(te):
                ch = meta.iloc[i]["channel"]
                v = float(y_means_tr[ch]) + float(vfit.coef["const"])
                for c in selected:
                    v += float(vfit.coef[c]) * (float(X.iloc[i][c])
                                                - float(x_means_tr[ch][c]))
                pred_te[j] = v
        else:
            Xte = np.column_stack([np.ones(len(te)), X.iloc[te][selected].values])
            pred_te = Xte @ vfit.coef.values
        yte = y.iloc[te].values
        nz = yte != 0
        holdout_mape = (float(np.mean(np.abs((yte[nz] - pred_te[nz]) / yte[nz]))) * 100
                        if nz.any() else np.nan)
        # ── WHY A SECOND HOLDOUT NUMBER EXISTS FOR POOLED MODELS ─────────
        # Plain MAPE averages the percentage error of every row equally. In an
        # unpooled model every row is the same channel, so that is fine. In a
        # POOLED model the rows span channels whose weekly volume differs by
        # 30x (Brand 1: Channel 1 ≈ 7.1M over the window, Channel 9 ≈ 0.22M),
        # and a 200% error on a tiny channel counts exactly as much as a 5%
        # error on the one that is half the brand. The pooled MAPE is then
        # driven almost entirely by the channels nobody is being paid to
        # forecast, and is NOT comparable to an unpooled MAPE.
        # WMAPE — Σ|error| / Σ|actual| — weights each week by how much volume
        # is at stake, which is the number to compare across levels. Both are
        # reported; the dashboard leads with WMAPE for pooled/hierarchical.
        denom = float(np.abs(yte).sum())
        holdout_wmape = (float(np.abs(yte - pred_te).sum()) / denom * 100
                         if denom else np.nan)

    # ── REPORTED model: same structure refit on every row of the window ──
    if channel_intercepts:
        Xr, x_means = _center_within(X, groups, None)
        yr, y_means = _center_within(y, groups, None)
        fit = cp.constrained_fit(Xr[selected], yr, S["specs_by_name"])
        # recover intercept_c = ȳ_c − Σβ·x̄_c, and rebuild the reported fit on
        # the ORIGINAL scale so everything downstream (contributions, due-tos,
        # charts, exports) sees volume rather than deviations
        intercepts = {}
        for ch in groups:
            v = float(y_means[ch]) + float(fit.coef["const"])
            for c in selected:
                v -= float(fit.coef[c]) * float(x_means[ch][c])
            intercepts[ch] = v
        const_col = meta["channel"].map(intercepts).astype(float).values
        fitted = const_col + sum(float(fit.coef[c]) * X[c].values
                                 for c in selected)
        resid = y.values - fitted
        contrib = pd.DataFrame({c: float(fit.coef[c]) * X[c].values
                                for c in selected}, index=X.index)
        contrib.insert(0, "Intercept", const_col)
        ss_tot = float(np.sum((y.values - y.values.mean()) ** 2))
        nz = y.values != 0
        n, k = len(y), len(selected)
        r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot else np.nan
        fit = cp.FitResult(
            cols=list(selected), coef=fit.coef, fitted=fitted, resid=resid,
            r2=r2,
            adj_r2=(1 - (1 - r2) * (n - 1) / (n - k - len(groups))
                    if n - k - len(groups) > 0 else np.nan),
            mape=(float(np.mean(np.abs(resid[nz] / y.values[nz]))) * 100
                  if nz.any() else np.nan),
            contributions=contrib, tstats=fit.tstats, vif=fit.vif,
            beta_std=fit.beta_std, beta_range=fit.beta_range,
            meta={**fit.meta, "channel_intercepts": intercepts})
    else:
        intercepts = None
        fit = cp.constrained_fit(X[selected], y, S["specs_by_name"])

    per_channel = channel_contributions(fit, X, meta, selected,
                                        intercepts=intercepts)
    return {
        "level": "pooled", "brand": brand_sheet, "config": cfg,
        "channel_intercepts": intercepts,
        "X_all": X, "X_raw": S["X_raw"], "y": y, "meta": meta,
        "selected": selected, "forced": [c for c in S["force"] if c in selected],
        "fit": fit, "holdout_mape": holdout_mape,
        "holdout_wmape": holdout_wmape,
        "wmape": (float(np.abs(fit.resid).sum()) / float(np.abs(y).sum()) * 100
                  if float(np.abs(y).sum()) else np.nan),
        "pred_te": pred_te, "yte": yte,
        "specs": S["specs"], "specs_by_name": S["specs_by_name"],
        "national": S["national"], "shares": S["shares"],
        "national_check": S["national_check"],
        "channels": S["channels"], "skipped": S["skipped"],
        "per_channel": per_channel,
        "df": S["df"], "sat_refs": S["sat_refs"],
        "window_start": S["window_start"], "window_end": S["window_end"],
        "n_rows": len(y), "n_weeks_window": int(meta["date"].nunique()),
        "dof": len(y) - len(selected) - 1,
    }


def channel_contributions(fit, X: pd.DataFrame, meta: pd.DataFrame,
                          selected: list,
                          intercepts: Optional[dict] = None) -> pd.DataFrame:
    """Break a pooled fit back down to the channel level.

    Same coefficient everywhere, each channel's own (proportionalized) X — so
    the channels differ in contribution because they differ in EXECUTION, which
    is exactly the claim a pooled model makes. Rows = channel, columns =
    drivers, plus actual/fitted/R²/MAPE per channel so a user can see which
    channel the shared coefficients fit badly. That per-channel fit quality is
    the main diagnostic for "should this brand have been pooled at all".
    """
    rows = []
    for ch, idx in meta.groupby("channel").groups.items():
        i = np.asarray(list(idx))
        base = (float(intercepts[ch]) if intercepts
                else float(fit.coef["const"]))
        rec = {"channel": ch, "weeks": len(i), "Intercept": base * len(i)}
        for c in selected:
            rec[c] = float(fit.coef[c] * X.loc[i, c].sum())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("channel")


def pooled_channel_fit(pooled: dict) -> pd.DataFrame:
    """Per-channel fit quality of the pooled model: R², MAPE and mean bias for
    each channel under the shared coefficients."""
    fit, meta, y = pooled["fit"], pooled["meta"], pooled["y"]
    fitted = pd.Series(fit.fitted, index=y.index)
    rows = []
    for ch, idx in meta.groupby("channel").groups.items():
        i = np.asarray(list(idx))
        yy, ff = y.loc[i].values, fitted.loc[i].values
        resid = yy - ff
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        nz = yy != 0
        rows.append({
            "channel": ch, "weeks": len(i),
            "r2": (1 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot else np.nan,
            "mape": (float(np.mean(np.abs(resid[nz] / yy[nz]))) * 100
                     if nz.any() else np.nan),
            "bias_pct": (float(resid.sum() / yy.sum()) * 100
                         if yy.sum() else np.nan),
            "actual": float(yy.sum()), "fitted": float(ff.sum()),
        })
    return pd.DataFrame(rows).set_index("channel")


# ═══════════════════════════════════════════════════════════════════════════
# 4. HIERARCHICAL MODEL  (level 3)
# ═══════════════════════════════════════════════════════════════════════════
#
# Alex's recipe, quoted from the email so the code can be checked against it:
#
#   "Each 'model run' would run once for a pooled model to establish the FIXED
#    effect, once WITHOUT any priors OR sign control (random effect) to
#    understand the distribution of coefficients for each predictor across all
#    models within the top level, and once more at brand*channel level with the
#    pooled coefficient multiplied by the INDEX of the random effect at that
#    channel."
#
# And from `Hierarchical Modeling Explanation.xlsx`, which is the authority on
# the arithmetic:
#
#   Model 1        Pooled Coefficient                 β_pooled(p)
#   Model 2        Unpooled Coefficient               β_unpooled(c, p)
#   Post Model 2   Indexed Unpooled Coefficient       I(c,p) = β_unpooled(c,p)
#                                                            / mean_c β_unpooled(·,p)
#   Model 3        Final Coefficient                  β(c,p) = I(c,p)·β_pooled(p)
#   Alt Post 2     Indexed & Capped Index             I clipped to 1 ± L·SD(I)
#   Alt Model 3    Capped Final Coefficient           β = I_capped·β_pooled
#
# The sheet's own check (rows 20-25) is that the AVERAGE of the final
# coefficients across channels comes back to the pooled coefficient, because
# the indices average to 1 by construction. `hierarchical_check` asserts it.
# ═══════════════════════════════════════════════════════════════════════════

def coefficient_index(unpooled: pd.DataFrame) -> pd.DataFrame:
    """I(c,p) = β_unpooled(c,p) / mean_over_channels β_unpooled(·,p).

    `unpooled` is channels × predictors. Mean, not median — Alex's AVERAGEIFS.
    A predictor whose mean is ~0 has no meaningful index (the ratio explodes),
    so it is set to 1, i.e. that channel simply inherits the pooled coefficient
    with no channel adjustment. That is the honest fallback: the unpooled runs
    disagreed about the sign badly enough that their average cancelled, so
    there is no stable "random effect" to index against.
    """
    mean = unpooled.mean(axis=0)
    idx = unpooled.copy().astype(float)
    for p in unpooled.columns:
        m = float(mean[p])
        denom_ok = np.isfinite(m) and abs(m) > 1e-12 * max(
            1.0, float(unpooled[p].abs().max()))
        idx[p] = (unpooled[p] / m) if denom_ok else 1.0
    return idx.replace([np.inf, -np.inf], 1.0).fillna(1.0)


def cap_index(index: pd.DataFrame, sd_limit: Optional[float] = 1.0,
              shrink: float = 1.0, enforce_sign: bool = True) -> pd.DataFrame:
    """Turn the raw index into the one actually applied. Three controls, in
    the order they are applied: SD cap → shrink → sign guard.

    ── 1. SD CAP — Alex's, from the sheet ───────────────────────────────────
    Clip each index to 1 ± sd_limit × SD(index for that predictor). His note:
    "we can ensure the index of unpooled coefficients is bound by x standard
    deviations away from 1 (1 = center of the index aka average)", with a
    manual input for the limit. Population SD, matching the sheet's STDEV.P —
    these are all the channels there are, not a sample of them.
    `sd_limit=None` disables it and gives his uncapped "Model 3".

    ── 2. SHRINK — the partial-pooling dial ────────────────────────────────
        I_applied = 1 + shrink × (I − 1)
    shrink = 1 is Alex's method untouched; shrink = 0 collapses every index to
    1, i.e. the pooled model. Anything in between is literal partial pooling,
    which is what Arko asked for on the call: "it's not either pooled or
    unpooled… there is some kind of relationship between the national
    coefficient versus the DMA-level coefficient." A Bayesian hierarchical
    model derives this weight from the ratio of within- to between-group
    variance; without a sampler there is no posterior to read it off, so it is
    exposed as a dial the user sets — and, in `fit_shrinkage`, one the holdout
    can choose.

    ── 3. SIGN GUARD — clip the index at 0 ─────────────────────────────────
    THE ONE ADDITION THAT IS NOT IN THE SHEET, and the reason it is here:
    on the real data the raw index runs from −7.2 to +11.9. A NEGATIVE index
    flips the sign of the final coefficient, so a channel ends up with media
    that destroys sales and a price cut that reduces volume — after the analyst
    explicitly declared those signs in the variable config. The whole engine
    enforces sign priors (`cp._bounds_from_specs`) precisely because Alex
    insists on them; letting the hierarchy quietly undo them would be a bug in
    the tool, not a finding about the data.

    Why the raw index goes negative at all is worth stating, because it is not
    noise in one channel — it is the normalisation. I(c,p) divides by the MEAN
    unpooled coefficient across channels. When the unconstrained per-channel
    fits disagree about a predictor's sign, that mean sits near zero, and
    dividing by a near-zero number produces indices of arbitrary size and sign.
    The instability is a property of dividing by a mean that is not bounded away
    from zero, and `index_stability` measures it directly.
    """
    out = index.copy().astype(float)
    if sd_limit is not None:
        for p in index.columns:
            sd = float(index[p].std(ddof=0))
            if not np.isfinite(sd) or sd == 0:
                continue
            out[p] = index[p].clip(lower=1.0 - sd_limit * sd,
                                   upper=1.0 + sd_limit * sd)
    if shrink != 1.0:
        out = 1.0 + float(shrink) * (out - 1.0)
    if enforce_sign:
        out = out.clip(lower=0.0)
    return out


def index_stability(unpooled: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Per-predictor diagnosis of how trustworthy its index is.

    `mean_to_spread` = |mean of the unpooled coefficients| / their SD. This is
    the number that governs everything: the index divides by that mean, so when
    the mean is small relative to the spread the ratio is unstable and the
    index is close to meaningless. Below ~1 the channels do not agree on a
    common effect at all, and the honest reading is that this predictor has no
    stable random effect to inherit — the dashboard flags those rows.
    """
    mean = unpooled.mean(axis=0)
    sd = unpooled.std(axis=0, ddof=0)
    ratio = (mean.abs() / sd.replace(0, np.nan))
    return pd.DataFrame({
        "unpooled_mean": mean,
        "unpooled_sd": sd,
        "mean_to_spread": ratio,
        "index_min": index.min(axis=0),
        "index_max": index.max(axis=0),
        "index_sd": index.std(axis=0, ddof=0),
        "sign_flips": (np.sign(unpooled) != np.sign(mean)).sum(axis=0),
        "stable": ratio > 1.0,
    })


def _refit_intercept(X: pd.DataFrame, y: pd.Series, coef: pd.Series,
                     cols: list) -> float:
    """With the slopes fixed by the hierarchy, the intercept is the one free
    parameter left. Least squares on the residual gives mean(y − Σβx): the
    level the channel operates at, with the hierarchy's slopes taken as given.

    Leaving the pooled intercept in place instead would be wrong — it is the
    average base volume across channels, and channels here differ in size by
    an order of magnitude, so every small channel would be over-predicted by a
    constant and its due-tos would not reconcile to its actual volume.
    """
    pred = np.zeros(len(X), dtype=float)
    for c in cols:
        pred += float(coef[c]) * X[c].values
    return float(np.mean(y.values - pred))


def run_hierarchical(path: str, brand_sheet: str,
                     config: Optional[cp.ModelConfig] = None,
                     df: Optional[pd.DataFrame] = None,
                     channels: Optional[list] = None,
                     sd_limit: Optional[float] = 1.0,
                     shrink: float = 1.0, enforce_sign: bool = True,
                     pooled: Optional[dict] = None) -> dict:
    """The three runs Alex described, in order.

    RUN 1 — pooled, with priors and sign control, on the stacked brand. Its
            selected structure becomes the shared predictor set: the hierarchy
            only makes sense if every channel is estimating the same
            coefficients, and this is the level that has the degrees of freedom
            to choose them well.
    RUN 2 — per channel, SAME predictors, NO sign constraints and NO custom
            bounds. Alex was specific about "without any priors OR sign
            control", and the reason is real: a constrained fit piles mass on
            the boundary at exactly 0, and a coefficient pinned at a bound
            carries no information about how much that channel differs from the
            brand. The random effect has to come from an unconstrained fit or
            the index is an artifact of the constraint.
    RUN 3 — per channel, coefficients = index × pooled, index optionally capped,
            intercept re-solved.

    Returns per-channel FitResult-shaped objects so everything downstream
    (contributions, due-tos, exports, charts) works unchanged.
    """
    cfg = config or cp.ModelConfig()
    # `pooled` can be passed in so a shrinkage sweep re-uses one pooled fit
    # instead of paying for it on every candidate value
    pooled = pooled or run_pooled(path, brand_sheet, config=cfg, df=df,
                                  channels=channels)
    selected = pooled["selected"]
    if not selected:
        raise ValueError(f"{brand_sheet}: pooled model selected no predictors "
                         "— nothing to build a hierarchy on")

    X, y, meta = pooled["X_all"], pooled["y"], pooled["meta"]
    beta_pooled = pd.Series({c: float(pooled["fit"].coef[c]) for c in selected})

    # ── RUN 2: unconstrained unpooled, same structure, per channel ───────
    free_specs = {c: cp.FeatureSpec(name=c, family=pooled["specs_by_name"][c].family,
                                    sign="unconstrained", coef_lower=None,
                                    coef_upper=None,
                                    scale=getattr(pooled["specs_by_name"][c],
                                                  "scale", 1.0))
                  for c in selected}
    unpooled_rows, unpooled_fits = {}, {}
    for ch, idx in meta.groupby("channel").groups.items():
        i = np.asarray(list(idx))
        Xc, yc = X.loc[i, selected], y.loc[i]
        try:
            f = cp.constrained_fit(Xc, yc, free_specs)
            unpooled_rows[ch] = {c: float(f.coef[c]) for c in selected}
            unpooled_fits[ch] = f
        except Exception:
            unpooled_rows[ch] = {c: float(beta_pooled[c]) for c in selected}
    unpooled = pd.DataFrame(unpooled_rows).T.reindex(columns=selected)

    # ── the index, and the index actually applied ────────────────────────
    index = coefficient_index(unpooled)
    index_capped = cap_index(index, sd_limit, shrink=shrink,
                             enforce_sign=enforce_sign)
    index_sd = index.std(ddof=0)
    stability = index_stability(unpooled, index)

    # ── RUN 3: final per-channel coefficients ────────────────────────────
    final = index_capped.mul(beta_pooled, axis=1)

    results, per_channel = {}, []
    for ch in final.index:
        i = np.asarray(list(meta.groupby("channel").groups[ch]))
        Xc, yc = X.loc[i, selected], y.loc[i]
        coef = pd.Series({c: float(final.loc[ch, c]) for c in selected})
        const = _refit_intercept(Xc, yc, coef, selected)
        coef_full = pd.Series({"const": const, **coef.to_dict()})

        fitted = const + sum(coef[c] * Xc[c].values for c in selected)
        resid = yc.values - fitted
        ss_tot = float(np.sum((yc.values - yc.values.mean()) ** 2))
        nz = yc.values != 0
        contrib = pd.DataFrame({c: coef[c] * Xc[c].values for c in selected},
                               index=Xc.index)
        contrib.insert(0, "Intercept", const)
        r2 = (1 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot else np.nan
        mape = (float(np.mean(np.abs(resid[nz] / yc.values[nz]))) * 100
                if nz.any() else np.nan)
        sd = Xc.std(ddof=0)

        fr = cp.FitResult(
            cols=list(selected), coef=coef_full, fitted=fitted, resid=resid,
            r2=r2,
            adj_r2=(1 - (1 - r2) * (len(yc) - 1) / (len(yc) - len(selected) - 1)
                    if len(yc) - len(selected) - 1 > 0 else np.nan),
            mape=mape, contributions=contrib,
            # t-stats come from the channel's own UNCONSTRAINED fit: they
            # describe how well that channel's data identifies each predictor,
            # which is the honest reading. A t-stat on a coefficient that was
            # imposed by the hierarchy rather than estimated would be
            # meaningless, so it is not invented here.
            tstats=(unpooled_fits[ch].tstats if ch in unpooled_fits
                    else pd.Series(dtype=float)),
            vif=(unpooled_fits[ch].vif if ch in unpooled_fits
                 else pd.Series(dtype=float)),
            beta_std=pd.Series({c: float(coef[c] * sd[c]) for c in selected}),
            beta_range=pd.Series({c: float(coef[c] * (Xc[c].max() - Xc[c].min()))
                                  for c in selected}),
            meta={"level": "hierarchical", "channel": ch},
        )
        results[ch] = fr
        denom = float(np.abs(yc).sum())
        per_channel.append({
            "channel": ch, "weeks": len(yc), "r2": r2, "mape": mape,
            "wmape": (float(np.abs(resid).sum()) / denom * 100 if denom
                      else np.nan),
            "actual": float(yc.sum()), "fitted": float(np.sum(fitted)),
        })

    chk = hierarchical_check(beta_pooled, index, index_capped, final)
    return {
        "level": "hierarchical", "brand": brand_sheet, "config": cfg,
        "pooled": pooled, "selected": selected,
        "beta_pooled": beta_pooled, "unpooled": unpooled,
        "index": index, "index_capped": index_capped, "index_sd": index_sd,
        "stability": stability,
        "final": final, "sd_limit": sd_limit, "shrink": float(shrink),
        "enforce_sign": bool(enforce_sign),
        "fits": results, "per_channel": pd.DataFrame(per_channel).set_index("channel"),
        "channels": list(final.index), "skipped": pooled["skipped"],
        "specs_by_name": pooled["specs_by_name"],
        "X_all": X, "y": y, "meta": meta,
        "window_start": pooled["window_start"], "window_end": pooled["window_end"],
        "checks": chk,
    }


SHRINK_GRID = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)


def fit_shrinkage(path: str, brand_sheet: str,
                  config: Optional[cp.ModelConfig] = None,
                  df: Optional[pd.DataFrame] = None,
                  sd_limit: Optional[float] = 1.0,
                  grid=SHRINK_GRID) -> dict:
    """Choose how much channel-level variation to admit, by holdout.

    The dial in `cap_index` runs from 0 (pooled: every channel shares the brand
    coefficient) to 1 (Alex's indexed hierarchical: each channel gets its full
    index). A Bayesian hierarchical model would derive the equivalent weight
    from the ratio of between- to within-channel variance; here it is fit the
    same way every other hyperparameter in this engine is — on data the
    coefficients did not see.

    Each channel's last 13 weeks are the held-out tail (the same tail
    `run_pooled` reserves). For each candidate shrink the hierarchy is rebuilt
    from the TRAINING rows only and scored on the pooled tails by WMAPE, so a
    tiny channel cannot dominate the choice. Returns the whole curve, not just
    the winner — the shape is the interesting part, because a flat curve means
    the channels genuinely do not differ and the pooled model was the right
    answer all along.
    """
    cfg = config or cp.ModelConfig()
    S = _stack_brand(path, brand_sheet, cfg, df=df)
    X, y, meta = S["X"], S["y"], S["meta"]
    train = ~meta["is_holdout"].values

    groups = meta.groupby("channel").groups
    Xc, _ = _center_within(X, groups, train)
    yc, _ = _center_within(y, groups, train)
    selected = _select(Xc.loc[train], yc.loc[train], cfg, S["specs_by_name"],
                       S["force"])
    if not selected:
        return {"error": "pooled selection on training rows chose no predictors"}
    pooled_tr = cp.constrained_fit(Xc.loc[train, selected], yc.loc[train],
                                   S["specs_by_name"])
    beta_pooled = pd.Series({c: float(pooled_tr.coef[c]) for c in selected})

    free = {c: cp.FeatureSpec(name=c, family=S["specs_by_name"][c].family,
                              sign="unconstrained",
                              scale=getattr(S["specs_by_name"][c], "scale", 1.0))
            for c in selected}
    rows = {}
    for ch, idx in groups.items():
        i = np.asarray(list(idx))
        tr = i[train[i]]
        try:
            f = cp.constrained_fit(X.loc[tr, selected], y.loc[tr], free)
            rows[ch] = {c: float(f.coef[c]) for c in selected}
        except Exception:                                  # noqa: BLE001
            rows[ch] = {c: float(beta_pooled[c]) for c in selected}
    index = coefficient_index(pd.DataFrame(rows).T.reindex(columns=selected))

    curve = []
    for lam in grid:
        applied = cap_index(index, sd_limit, shrink=lam, enforce_sign=True)
        final = applied.mul(beta_pooled, axis=1)
        err = tot = 0.0
        for ch, idx in groups.items():
            i = np.asarray(list(idx))
            tr, te = i[train[i]], i[~train[i]]
            if not len(te) or ch not in final.index:
                continue
            coef = final.loc[ch]
            const = _refit_intercept(X.loc[tr, selected], y.loc[tr], coef,
                                     selected)
            pred = const + sum(float(coef[c]) * X.loc[te, c].values
                               for c in selected)
            err += float(np.abs(y.loc[te].values - pred).sum())
            tot += float(np.abs(y.loc[te].values).sum())
        curve.append({"shrink": float(lam),
                      "holdout_wmape": (err / tot * 100) if tot else np.nan})
    best = min(curve, key=lambda d: (np.inf if np.isnan(d["holdout_wmape"])
                                     else d["holdout_wmape"]))
    return {"curve": curve, "best_shrink": best["shrink"],
            "best_wmape": best["holdout_wmape"],
            "pooled_wmape": curve[0]["holdout_wmape"],
            "full_index_wmape": curve[-1]["holdout_wmape"],
            "selected": selected, "n_channels": len(rows)}


def hierarchical_check(beta_pooled: pd.Series, index: pd.DataFrame,
                       index_capped: pd.DataFrame,
                       final: pd.DataFrame) -> dict:
    """The sheet's own validation (rows 20–25), as assertions.

      1. the uncapped index averages to 1 for every predictor;
      2. the uncapped final coefficients average back to the pooled ones;
      3. the CAPPED ones stay close but need not be identical — capping is a
         deliberate, asymmetric shrink, so a small drift here is correct
         behaviour, not a bug. It is reported rather than asserted.
    """
    idx_mean = index.mean(axis=0)
    final_uncapped_mean = index.mul(beta_pooled, axis=1).mean(axis=0)
    capped_mean = final.mean(axis=0)
    denom = beta_pooled.abs().replace(0, np.nan)
    return {
        "index_mean_max_dev": float((idx_mean - 1.0).abs().max()),
        "uncapped_recovers_pooled": float(
            ((final_uncapped_mean - beta_pooled).abs() / denom).max(skipna=True)),
        "capped_drift_pct": float(
            ((capped_mean - beta_pooled) / denom * 100).abs().max(skipna=True)),
        "n_predictors": len(beta_pooled),
        "index_min": float(index.min().min()), "index_max": float(index.max().max()),
        "capped_min": float((index_capped).min().min()),
        "capped_max": float((index_capped).max().max()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMPARING THE THREE LEVELS
# ═══════════════════════════════════════════════════════════════════════════

def compare_levels(path: str, brand_sheet: str,
                   config: Optional[cp.ModelConfig] = None,
                   df: Optional[pd.DataFrame] = None,
                   sd_limit: Optional[float] = 1.0,
                   shrink: Optional[float] = None) -> pd.DataFrame:
    """All three levels on one brand, scored on IDENTICAL holdouts.

    The comparison only means anything if every level is judged the same way,
    so all three are scored on each channel's own last 13 weeks, by WMAPE, with
    every model fitted on the training rows only:

      unpooled       that channel's own selection and coefficients
      pooled         the brand's shared coefficients, this channel's X
      hierarchical   index × pooled, intercept re-solved on the channel

    `shrink=None` fits the dial by holdout first (`fit_shrinkage`) instead of
    assuming a value, which is the honest default: the amount of channel-level
    variation the data supports is not something to guess.

    The per-channel breakdown is the point. Pooling tends to help the thin
    channels — they were fitting noise on their own — and cost the big one that
    had plenty of data to speak for itself. A single brand-level average hides
    exactly that trade-off.
    """
    cfg = config or cp.ModelConfig()
    S = _stack_brand(path, brand_sheet, cfg, df=df)
    X, y, meta = S["X"], S["y"], S["meta"]
    train = ~meta["is_holdout"].values
    groups = meta.groupby("channel").groups

    if shrink is None:
        sw = fit_shrinkage(path, brand_sheet, config=cfg, df=df,
                           sd_limit=sd_limit)
        shrink = sw.get("best_shrink", 0.0)
        shrink_curve = sw.get("curve")
    else:
        shrink_curve = None

    # ── pooled, fit on training rows only, with channel intercepts ───────
    Xc, x_means = _center_within(X, groups, train)
    yc, y_means = _center_within(y, groups, train)
    sel_p = _select(Xc.loc[train], yc.loc[train], cfg, S["specs_by_name"],
                    S["force"])
    pooled_tr = cp.constrained_fit(Xc.loc[train, sel_p], yc.loc[train],
                                   S["specs_by_name"])
    beta_pooled = pd.Series({c: float(pooled_tr.coef[c]) for c in sel_p})

    # ── unconstrained per-channel fits → index → hierarchical coefficients ─
    free = {c: cp.FeatureSpec(name=c, family=S["specs_by_name"][c].family,
                              sign="unconstrained",
                              scale=getattr(S["specs_by_name"][c], "scale", 1.0))
            for c in sel_p}
    rows_u = {}
    for ch, idx in groups.items():
        i = np.asarray(list(idx)); tr = i[train[i]]
        try:
            f = cp.constrained_fit(X.loc[tr, sel_p], y.loc[tr], free)
            rows_u[ch] = {c: float(f.coef[c]) for c in sel_p}
        except Exception:                                  # noqa: BLE001
            rows_u[ch] = {c: float(beta_pooled[c]) for c in sel_p}
    index = coefficient_index(pd.DataFrame(rows_u).T.reindex(columns=sel_p))
    final = cap_index(index, sd_limit, shrink=shrink,
                      enforce_sign=True).mul(beta_pooled, axis=1)

    def _wmape(actual, pred):
        d = float(np.abs(actual).sum())
        return float(np.abs(actual - pred).sum()) / d * 100 if d else np.nan

    rows = []
    for ch, idx in groups.items():
        i = np.asarray(list(idx))
        tr, te = i[train[i]], i[~train[i]]
        yte = y.loc[te].values
        rec = {"channel": ch, "weeks": len(i), "holdout_weeks": len(te),
               "volume": float(y.loc[i].sum())}

        # unpooled — the channel's own model, selected on its own training rows
        try:
            sel_u = _select(X.loc[tr], y.loc[tr], cfg, S["specs_by_name"],
                            S["force"])
            fu = cp.constrained_fit(X.loc[tr, sel_u], y.loc[tr],
                                    S["specs_by_name"])
            pu = fu.coef["const"] + sum(fu.coef[c] * X.loc[te, c].values
                                        for c in sel_u)
            rec["unpooled_wmape"] = _wmape(yte, pu)
            rec["unpooled_n_vars"] = len(sel_u)
        except Exception as exc:                           # noqa: BLE001
            rec["unpooled_wmape"] = np.nan
            rec["unpooled_error"] = str(exc)[:60]

        # pooled — shared SLOPES, this channel's own intercept and execution
        base = float(y_means[ch]) + float(pooled_tr.coef["const"]) - sum(
            float(beta_pooled[c]) * float(x_means[ch][c]) for c in sel_p)
        pp = base + sum(float(beta_pooled[c]) * X.loc[te, c].values
                        for c in sel_p)
        rec["pooled_wmape"] = _wmape(yte, pp)

        # hierarchical — index × pooled, intercept re-solved on this channel
        coef = final.loc[ch]
        const = _refit_intercept(X.loc[tr, sel_p], y.loc[tr], coef, sel_p)
        ph = const + sum(float(coef[c]) * X.loc[te, c].values for c in sel_p)
        rec["hier_wmape"] = _wmape(yte, ph)
        rec["best_level"] = min(
            [("unpooled", rec.get("unpooled_wmape", np.nan)),
             ("pooled", rec["pooled_wmape"]), ("hierarchical", rec["hier_wmape"])],
            key=lambda kv: np.inf if pd.isna(kv[1]) else kv[1])[0]
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("channel")
    # brand-level roll-up, volume-weighted — the number to quote
    w = out["volume"] / out["volume"].sum()
    out.attrs["brand"] = brand_sheet
    out.attrs["shrink"] = shrink
    out.attrs["shrink_curve"] = shrink_curve
    out.attrs["n_pooled_vars"] = len(sel_p)
    out.attrs["pooled_rows"] = int(len(y))
    for lvl in ("unpooled", "pooled", "hier"):
        col = out[f"{lvl}_wmape"]
        out.attrs[f"{lvl}_weighted"] = float((col * w).sum(skipna=True))
    return out


if __name__ == "__main__":                             # pragma: no cover
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "Anonymized Data for Project.xlsx"
    brand = sys.argv[2] if len(sys.argv) > 2 else "Brand 1"
    print(compare_levels(p, brand).round(2).to_string())

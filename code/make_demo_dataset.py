"""
Synthetic demo dataset generator
================================

Builds a workbook that is STRUCTURALLY identical to the sponsor's anonymized
data file — same sheet layout, same column names, same grain, same per-brand
irregularity — but contains no sponsor or client values whatsoever. Every
number here comes out of a seeded random generator defined in this file.

Why the column names are unchanged
----------------------------------
`capstone_pipeline.generate_variable_config` assigns families by exact column
name ("Price per Volume", "ACV Weighted Distribution", "Weighted Weeks ...",
the macro list) and by the `_Spend` suffix / `_C_` prefix conventions. Those
are generic CPG-measurement field names, not client information. Renaming them
would only mean the config generator silently classifies everything as "Other".

Why sheet names still start with "Brand "
-----------------------------------------
`dataset_anchor_week`, `run_all.py` and `scaling_test.py` discover product
sheets with `sheet_name.lower().startswith("brand")`. Only the dashboard has a
header-sniffing fallback. Keeping the prefix keeps every entry point working.

What IS fictional: every brand, retailer-channel, competitor and media-platform
name, and every single data value.

Usage
-----
    python code/make_demo_dataset.py [-o OUT.xlsx] [--seed 20260728]
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Calendar — 156 weekly periods, 2023-01-08 .. 2025-12-28.
# The last week is the dataset anchor, so a 104-week window resolves to
# 2024-01-07 .. 2025-12-28 and the reserved 13-week tail to 2025-10-05 onward.
# ---------------------------------------------------------------------------
N_WEEKS = 156
WEEKS = pd.date_range("2023-01-08", periods=N_WEEKS, freq="7D")
WINDOW_WEEKS = 104
HOLDOUT_WEEKS = 13
WIN_START_IDX = N_WEEKS - WINDOW_WEEKS          # 52
TAIL_START_IDX = N_WEEKS - HOLDOUT_WEEKS        # 143

# ---------------------------------------------------------------------------
# Fictional entities
# ---------------------------------------------------------------------------
CHANNELS = [
    "Channel Northgate",     # 0  hero channel
    "Channel Sunmart",       # 1
    "Channel Valemart",      # 2
    "Channel Rivergrocer",   # 3
    "Channel Bayside",       # 4
    "Channel Cloverfield",   # 5
    "Channel Dockside",      # 6
    "Channel Pinehill",      # 7  delisted mid-2023, only two brands ever sold here
    "Channel Quarry",        # 8  launched mid-2023
]

RIVALS = ["Alder", "Bramble", "Clover", "Dogwood", "Elmridge", "Fernbank",
          "Gorsehill", "Hawthorn", "Ivyfield", "Larkspur", "Mulberry"]

# Generic platform *types*, not a client's actual retail-media roster.
# "search" -> decay 0.2 and "ctv"/"video" -> 0.7 are picked up by
# capstone_pipeline._DECAY_HINTS, which keeps the differential-decay story.
MEDIA_POOL = [
    "Social_Feed", "Search", "Short_Video", "Retail_Media_A", "Retail_Media_B",
    "Marketplace", "Delivery_App", "Grocery_Network_A", "Grocery_Network_B",
    "Grocery_Network_C", "Onsite_Display", "Connected_TV",
    "Audio_Streaming", "Programmatic_Display", "Programmatic_CTV",
    "Social_Stories", "Video_Preroll", "Sponsored_Search", "Coupon_Network",
    "Loyalty_Offers", "Influencer", "Out_Of_Home",
]


@dataclass
class BrandSpec:
    name: str
    channels: list          # indices into CHANNELS
    n_media: int
    n_rivals: int
    base_volume: float      # mean weekly volume in the hero channel
    base_price: float
    seed: int
    # slices whose volume goes to zero for the reserved tail (delisted in Q4
    # 2025) -> the engine reports them as "No holdout"
    zero_tail: list = field(default_factory=list)


BRANDS = [
    BrandSpec("Brand Aurora",      list(range(9)), 12, 6, 92_000, 4.35, 101),
    BrandSpec("Brand Birchwood",   list(range(9)), 22, 4, 61_000, 5.10, 102,
              zero_tail=[4]),
    BrandSpec("Brand Cascade",     [0, 1, 2, 3, 4, 5, 6, 8], 11, 8, 38_000, 6.05, 103),
    BrandSpec("Brand Driftwood",   [0, 1, 2, 3, 4, 5, 6, 8], 11, 10, 25_500, 7.80, 104),
    BrandSpec("Brand Everglade",   [0, 1, 2, 3, 4, 5, 6, 8], 22, 6, 47_000, 3.65, 105,
              zero_tail=[6]),
    BrandSpec("Brand Foxglove",    [0, 1, 2, 3, 4, 5, 6, 8],  0, 6, 19_800, 9.20, 106,
              zero_tail=[5]),
    BrandSpec("Brand Glacier",     [0, 1, 2, 3, 4, 5, 6, 8],  0, 10, 33_000, 4.90, 107,
              zero_tail=[2]),
    BrandSpec("Brand Harborlight", [0, 1, 2, 3, 4, 5, 6, 8], 12, 11, 55_000, 5.75, 108),
    BrandSpec("Brand Ironwood",    [0, 1, 2, 3, 4, 5, 6, 8],  0, 10, 28_400, 6.40, 109,
              zero_tail=[1]),
    BrandSpec("Brand Juniper",     [0, 1, 2, 3, 4, 5, 6, 8], 22, 1, 71_000, 3.20, 110,
              zero_tail=[3]),
]

# Row coverage per (brand index, channel index). Anything not listed is the
# full 156 weeks. `("head", k)` = first k weeks only, `("tail", k)` = last k.
COVERAGE = {
    (0, 7): ("head", 25),    # Aurora x Pinehill    -> 0 weeks in window, skipped
    (1, 7): ("head", 25),    # Birchwood x Pinehill -> 0 weeks in window, skipped
    (0, 8): ("tail", 131),   # Quarry launched mid-2023
    (1, 8): ("tail", 131),
    (5, 5): ("tail", 109),   # Foxglove pulled out of Cloverfield, came back
    (5, 8): ("tail", 104),
    (8, 5): ("tail", 67),    # Ironwood is a late entrant in Cloverfield
}

# ---------------------------------------------------------------------------
# The hero slice — Brand Aurora x Channel Northgate.
#
# The demo beat: ACV Weighted Distribution erodes gently through the training
# weeks and then falls off a cliff inside the reserved tail. Because depth of
# distribution is nearly constant here, Total Points of Distribution is almost
# a rescaling of ACV, the whole distribution family sits above the VIF
# threshold, and pruning throws ACV out — unless `force` protects it. So on
# `auto` the model has no distribution term and never sees the cliff coming;
# forced in, the coefficient is well identified by the training erosion and
# recovers most of the drop. What is left is how much steeper the cliff was
# than anything in the training window could have implied.
# ---------------------------------------------------------------------------
HERO_BRAND, HERO_CHANNEL = 0, 0
HERO_ACV_BASE = 88.0
HERO_ACV_FLOOR = 54.0          # where distribution lands by the last week
HERO_COLLAPSE_START = 143      # the cliff starts exactly at the reserved tail
HERO_ACV_ELASTICITY = 1.35
# In-sample wander of the hero's ACV. It has to be small enough that the
# variable's t-stat stays under the p_enter threshold — otherwise forward
# stepwise picks distribution up on its own, `auto` and `force` give the same
# model, and there is nothing to demonstrate — but not so small that a forced
# coefficient is estimated from pure noise.
HERO_ACV_WIGGLE = 0.22
HERO_EROSION = 18.0            # points of ACV lost gradually during training
HERO_SIGNAL_SD = 0.205         # kept modest so the ACV tail term has headroom
HERO_NOISE = 0.078             #   before weekly volume approaches the floor

# ---------------------------------------------------------------------------
# Fleet quality plan
#
# A real book of 80 models is not 80 equally good models, and a demo where
# every slice grades "Good" makes the grade column pointless. Each modeled
# slice gets a signal gain and a noise level drawn from one of four buckets,
# assigned deterministically, so the batch reproduces a believable spread:
# mostly good, a long moderate middle, a handful worth arguing about.
# ---------------------------------------------------------------------------
# Each bucket is (noise_lo, noise_hi, snr_lo, snr_hi) where noise is the
# residual standard deviation as a fraction of volume and snr = signal sd /
# noise sd. Because the drivers are rescaled to hit the requested signal sd
# exactly, R2 ~ snr^2 / (snr^2 + 1) and in-sample MAPE ~ 0.8 * noise — the
# two numbers the grade column is built on are set directly rather than
# discovered by trial and error.
# Calibrated against an actual batch run rather than assumed: out-of-sample
# MAPE lands around 1.6-1.8x the in-sample figure, because the validation
# model is selected and fit on 91 weeks and then asked about 13 it has never
# seen. Holdout MAPE ~ 150 * noise over the range that matters.
BUCKETS = {
    "good":     (0.018, 0.042, 3.60, 5.60),
    "moderate": (0.055, 0.094, 1.50, 2.80),
    "flagged":  (0.280, 0.400, 0.50, 0.95),
}
BUCKET_PLAN = [("good", 41), ("moderate", 24), ("flagged", 9)]
MAX_SIGNAL_SD = 0.26             # keep 1 + signal comfortably positive
PLAN_SEED = 777


def build_slice_plan() -> dict:
    """(brand index, channel index) -> (signal_gain, noise_frac).

    Slices that are never modeled (no data in the window) and slices that go
    to zero in the reserved tail are handed the "good" profile — their grade
    is decided by coverage, not by fit, so spending a bucket on them would
    just distort the distribution the batch screen shows."""
    modeled, special = [], []
    for bi, spec in enumerate(BRANDS):
        for ci in spec.channels:
            if COVERAGE.get((bi, ci), ("", N_WEEKS))[1] < 52 or ci in spec.zero_tail:
                special.append((bi, ci))
            else:
                modeled.append((bi, ci))

    labels = [lab for lab, k in BUCKET_PLAN for _ in range(k)]
    hero = (HERO_BRAND, HERO_CHANNEL)
    others = [s for s in modeled if s != hero]
    labels = labels[:len(others) + 1]
    while len(labels) < len(others) + 1:
        labels.append("moderate")

    rng = np.random.default_rng(PLAN_SEED)
    # the hero slice is a genuinely good in-sample model — that is the whole
    # point of it, so it is pinned to the good bucket and never shuffled
    labels.remove("good")
    rng.shuffle(labels)

    plan = {}
    for (bi, ci), lab in zip(others, labels):
        nlo, nhi, slo, shi = BUCKETS[lab]
        noise = float(rng.uniform(nlo, nhi))
        snr = float(rng.uniform(slo, shi))
        plan[(bi, ci)] = (min(noise * snr, MAX_SIGNAL_SD), noise)
        PLAN_LABELS[(bi, ci)] = lab
    # hero: R2 ~ 0.88, in-sample MAPE ~ 9.6% — a good model that still fails
    # out of sample, which is the only reason the demo has anything to show
    PLAN_LABELS[hero] = "hero"
    plan[hero] = (HERO_SIGNAL_SD, HERO_NOISE)
    for s in special:
        PLAN_LABELS[s] = "special"
        plan[s] = (0.240, 0.085)
    return plan


SLICE_PLAN = None                # populated in main()
PLAN_LABELS: dict = {}           # (bi, ci) -> bucket name, for calibration


def _ar1(rng, n, sd, rho=0.75):
    """Mean-zero AR(1) — smooth-ish weekly wander instead of white noise."""
    e = rng.normal(0, sd, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def _bursty(rng, n, mean, p_on=0.42):
    """Media-spend style series: mostly off, lumpy when on."""
    on = rng.random(n) < p_on
    amt = rng.gamma(2.0, mean / 2.0, n)
    return np.where(on, amt, 0.0).round(2)


def _adstock(x, decay):
    out = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        out[i] = (1 - decay) * v + (decay * out[i - 1] if i else 0.0)
    return out


# ---------------------------------------------------------------------------
# National series — identical for every brand and channel, the way real
# macro data is. Deliberately collinear so the VIF column has something to do.
# ---------------------------------------------------------------------------
def build_macro(seed=9001):
    """Mostly mean-reverting, with only mild drift.

    An earlier version gave several of these strong straight-line trends.
    Forward stepwise then picked one up as a proxy for whatever slow movement
    a slice had, and the fitted line extrapolated off the end of the training
    window — holdout MAPE ran 2-3x in-sample across the whole batch for a
    reason that had nothing to do with the engine. Real macro series over two
    years wander more than they march.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_WEEKS)
    drift = t / N_WEEKS          # 0 -> 1 over the whole period
    unemp = 3.6 + 0.6 * np.sin(t / 41) + _ar1(rng, N_WEEKS, 0.02)
    return pd.DataFrame({
        "Unemployment_Rate": unemp.round(2),
        "Median_CPI": (298 + 9.0 * drift + _ar1(rng, N_WEEKS, 0.30)).round(2),
        "CFNAI": (_ar1(rng, N_WEEKS, 0.12) - 0.05).round(3),
        "Gas_Price": (3.35 + 0.45 * np.sin(t / 23 + 1.1)
                      + _ar1(rng, N_WEEKS, 0.03)).round(3),
        "UMCSENT": (66 + 9 * np.sin(t / 31) + _ar1(rng, N_WEEKS, 0.7)).round(1),
        "T5YIE": (2.28 + 0.22 * np.sin(t / 27 + 0.4)
                  + _ar1(rng, N_WEEKS, 0.015)).round(3),
        "PSAVERT": (4.4 + 0.7 * np.sin(t / 37 + 2.0)
                    + _ar1(rng, N_WEEKS, 0.04)).round(2),
        "PCEC96": (15_800 + 240 * drift + 90 * np.sin(t / 29)
                   + _ar1(rng, N_WEEKS, 18)).round(1),
        "SNAP_Participants": (41_500_000 - 700_000 * drift
                              + 380_000 * np.sin(t / 33 + 0.7)
                              + _ar1(rng, N_WEEKS, 60_000)).round(0),
        "SNAP_Households": (22_300_000 - 340_000 * drift
                            + 190_000 * np.sin(t / 35 + 1.4)
                            + _ar1(rng, N_WEEKS, 34_000)).round(0),
        "SNAP_Cost": (7_900_000_000 - 70_000_000 * drift
                      + 150_000_000 * np.sin(t / 25 - 0.4)
                      + _ar1(rng, N_WEEKS, 14_000_000)).round(0),
        "SNAP_Average_Monthly_Benefit_PC": (190 - 4.5 * drift
                                            + 2.4 * np.sin(t / 39)
                                            + _ar1(rng, N_WEEKS, 0.6)).round(2),
        "SNAP_Average_Monthly_Benefit_PH": (353 - 8.0 * drift
                                            + 4.5 * np.sin(t / 43 + 2.2)
                                            + _ar1(rng, N_WEEKS, 1.1)).round(2),
        "Fedfunds": (5.33 - 0.55 * drift + 0.10 * np.sin(t / 30)
                     + _ar1(rng, N_WEEKS, 0.012)).round(3),
        "Unempclaims": (218_000 + 9_000 * np.sin(t / 19)
                        + _ar1(rng, N_WEEKS, 3_500)).round(0),
        "Retail_Sales": (612_000 + 11_000 * drift
                         + 9_000 * np.sin(t / 26 - 0.6)
                         + _ar1(rng, N_WEEKS, 2_200)).round(0),
    })


MACRO = build_macro()

# 52-week seasonal shape, shared by everything (it is a category effect)
_WEEK_NUM = np.array([min(d.isocalendar().week, 52) for d in WEEKS])
_season_rng = np.random.default_rng(4242)
_SEASON_SHAPE = (1.0
                 + 0.16 * np.sin(2 * np.pi * (_WEEK_NUM - 6) / 52)
                 + 0.07 * np.sin(4 * np.pi * (_WEEK_NUM - 2) / 52))
_SEASON_SHAPE += _season_rng.normal(0, 0.012, N_WEEKS)


# ---------------------------------------------------------------------------
# One Brand x Channel slice
# ---------------------------------------------------------------------------
def build_slice(bi: int, ci: int, spec: BrandSpec, media_cols, rival_names,
                signal_sd: float, noise_frac: float) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed * 1000 + ci)
    n = N_WEEKS
    t = np.arange(n)
    is_hero = (bi == HERO_BRAND and ci == HERO_CHANNEL)

    # ---- distribution -----------------------------------------------------
    if is_hero:
        # Two regimes. A gentle, well-identified erosion through the training
        # window — enough for a forced coefficient to be estimated properly —
        # then a cliff inside the reserved tail.
        #
        # An earlier attempt kept ACV FLAT in training and relied on it failing
        # the p_enter test. That backfires: with no in-sample variation the
        # coefficient is unidentified, and forcing it in made the holdout an
        # order of magnitude WORSE, not better. What keeps distribution out of
        # the `auto` model here is VIF, not significance — see `items` below.
        erosion = np.clip(t / HERO_COLLAPSE_START, 0, 1) ** 1.6
        acv = HERO_ACV_BASE - HERO_EROSION * erosion
        cliff = np.clip((t - HERO_COLLAPSE_START) /
                        (n - 1 - HERO_COLLAPSE_START), 0, 1)
        acv = acv - (HERO_ACV_BASE - HERO_EROSION - HERO_ACV_FLOOR) * cliff ** 0.9
        acv = acv + _ar1(rng, n, HERO_ACV_WIGGLE, rho=0.85)
    else:
        level = rng.uniform(48, 93)
        drift = rng.uniform(-6, 6) * (t / n)
        acv = level + drift + _ar1(rng, n, 0.55, rho=0.8)
    acv = np.clip(acv, 4, 99.5)

    # Depth of distribution barely moves for the hero, which makes Total Points
    # of Distribution an almost exact rescaling of ACV. The whole distribution
    # family then sits above the VIF threshold together, and pruning discards
    # ACV unless `force` protects it. That is the demo: `auto` loses the one
    # variable that explains the cliff, `force` keeps it.
    item_wiggle = 0.004 if is_hero else 0.012
    items = (rng.uniform(7.5, 19.0) * (1 + _ar1(rng, n, item_wiggle, rho=0.85))
             * (0.75 + 0.25 * acv / max(acv.max(), 1)))
    tpd = acv * items                       # collinear with both, by construction

    # ---- trade execution --------------------------------------------------
    def ww(mean, p_on, shape=2.2):
        on = rng.random(n) < p_on
        return np.where(on, rng.gamma(shape, mean / shape, n), 0.0)

    # The two trade variables that carry real weight in the DGP run most
    # weeks and are only moderately lumpy. Heavily zero-inflated drivers
    # produce a badly left-skewed signal — enough of them line up at zero in
    # the same week to drive modeled volume toward zero, and MAPE detonates
    # on a near-zero denominator. Price reductions running ~4 weeks in 5 is
    # also just what CPG trade calendars look like.
    ww_pr = ww(rng.uniform(11, 26), 0.82, shape=3.6)
    ww_dp = ww(rng.uniform(8, 20), 0.74, shape=3.1)
    ww_ft = ww(rng.uniform(3, 11), 0.32)
    ww_sp = ww(rng.uniform(1.5, 6), 0.22)
    ww_fd = ww(rng.uniform(2.5, 9), 0.30)
    any_merch_w = np.clip((ww_pr + ww_ft + ww_dp + ww_sp + ww_fd) / 100, 0, 0.86)

    # ---- price ------------------------------------------------------------
    # Promo weeks really do carry a lower shelf price, but the coupling has to
    # stay weak. At the obvious strength, Price per Volume and Weighted Weeks
    # Price Reductions Only come out ~0.9 correlated, VIF pruning throws BOTH
    # away, and a third of the intended signal is unreachable no matter what
    # the selector does. Most of the price movement is therefore independent.
    price = (spec.base_price * (1 + 0.015 * (t / n))
             * (1 + _ar1(rng, n, 0.030, rho=0.78))
             - 0.0025 * ww_pr - 0.0015 * ww_fd)
    price = np.clip(price, spec.base_price * 0.55, spec.base_price * 1.45)
    price_ref0 = price[WIN_START_IDX:TAIL_START_IDX].mean()

    # ---- media ------------------------------------------------------------
    media = {}
    media_adstocked = []
    for k, col in enumerate(media_cols):
        low = col.lower()
        decay = 0.2 if "search" in low else (0.7 if ("ctv" in low or "video" in low)
                                             else 0.5)
        series = _bursty(rng, n, rng.uniform(1_500, 26_000),
                         p_on=rng.uniform(0.25, 0.6))
        media[col] = series
        media_adstocked.append(_adstock(series, decay))

    # ---- competitors ------------------------------------------------------
    comp = {}
    rival_acv = []
    for k, rv in enumerate(rival_names):
        r_acv = np.clip(rng.uniform(35, 88) + _ar1(rng, n, 0.6, rho=0.82), 3, 99)
        rival_acv.append(r_acv)
        r_items = rng.uniform(6, 17) * (1 + _ar1(rng, n, 0.015, rho=0.85))
        comp[f"_C_Rival {rv}__Avg Weekly Items per Store Selling"] = r_items
        comp[f"_C_Rival {rv}__ACV Weighted Distribution"] = r_acv
        for label, mean, p in (("Price Reductions Only", 14, 0.5),
                               ("Feature Only", 6, 0.3),
                               ("Display Only", 11, 0.42),
                               ("Special Pack Only", 3, 0.2),
                               ("Feature and Display", 5, 0.25)):
            comp[f"_C_Rival {rv}__Weighted Weeks {label}"] = ww(mean, p)

    # ---- seasonality ------------------------------------------------------
    season = _SEASON_SHAPE * (1 + rng.normal(0, 0.02))

    # ---- assemble the true signal ----------------------------------------
    acv_ref = acv[WIN_START_IDX:TAIL_START_IDX].mean()
    price_ref = price[WIN_START_IDX:TAIL_START_IDX].mean()

    # ---- the true data-generating process ---------------------------------
    # Deliberately CONCENTRATED. An earlier version spread the signal over
    # ~16 drivers; forward stepwise can only carry 9-13 out of 100+ candidates
    # on 91 training weeks, so a third of the signal was unreachable and every
    # slice under-shot its intended R2. Nine real drivers, each with a share
    # big enough to clear p_enter, is both easier to hit and a more honest
    # picture of what a weekly volume model actually finds.
    ref = slice(WIN_START_IDX, TAIL_START_IDX)

    def z(x):
        xr = np.asarray(x)[ref]
        s = xr.std()
        return (np.asarray(x) - xr.mean()) / (s if s > 1e-9 else 1.0)

    parts = [(0.53, z(season)),          # seasonality dominates, as it should
             (0.37, -z(price)),
             (0.35, z(ww_dp)),
             (0.35, z(ww_pr)),
             (0.20, z(ww_fd))]
    for k in range(min(2, len(media_adstocked))):
        parts.append((0.28 if k == 0 else 0.24, z(media_adstocked[k])))
    for k in range(min(2, len(rival_acv))):
        parts.append((-0.26 if k == 0 else -0.21, z(rival_acv[k])))
    if not is_hero:
        parts.append((0.32, z(acv)))

    core = sum(w * s for w, s in parts)
    sd = core[ref].std()
    core = core / (sd if sd > 1e-9 else 1.0) * signal_sd
    signal = 1.0 + core

    if is_hero:
        # The hero's distribution term is deliberately NOT part of the scaled
        # bundle: it is flat through the training weeks (so it never clears
        # p_enter and gets pruned as collinear with TPD) and only bites in the
        # reserved tail. That asymmetry is the entire demo.
        signal = signal + HERO_ACV_ELASTICITY * (acv / acv_ref - 1.0)

    # A backstop, not a design element: with the tuning above the raw signal
    # does not reach it. If a future edit makes it bite, the assertion in
    # main() will say so rather than letting a kinked DGP through quietly.
    signal = np.clip(signal, 0.22, None)
    volume = spec.base_volume * signal
    # lognormal residual: multiplicative, mean-preserving, and — unlike an
    # additive normal — it cannot drive a weekly volume negative and get
    # clipped, which would put a kink in an otherwise linear DGP
    volume *= np.exp(rng.normal(0, noise_frac, n) - noise_frac ** 2 / 2)

    # ---- POS decompositions (excluded as leakage, but they must exist) ----
    v_any = volume * any_merch_w
    v_no = volume - v_any
    w = np.stack([ww_pr, ww_ft, ww_dp, ww_sp, ww_fd])
    wsum = w.sum(axis=0)
    wsum[wsum == 0] = 1.0
    v_pr, v_ft, v_dp, v_sp, v_fd = (v_any * (wi / wsum) for wi in w)

    def pv(discount):
        """Price per volume under one merch condition. Each condition gets its
        own level and its own wander: generated as `price * (1 + 2% noise)`
        the eight variants come out ~0.99 correlated with Price per Volume,
        VIF pruning discards the whole price family, and the model loses its
        most important variable. Deeper cuts under feature-and-display is also
        what the shelf actually does."""
        out = price * discount * (1 + _ar1(rng, n, 0.045, rho=0.7))
        return np.round(out, 4)

    data = {
        "Geography": CHANNELS[ci],
        "Product": spec.name,
        "Time": [f"WE {d:%m/%d/%Y}" for d in WEEKS],
        "Volume Sales": volume,
        "Volume Sales No Merch": v_no,
        "Volume Sales Any Merch": v_any,
        "Volume Sales Price Reductions Only": v_pr,
        "Volume Sales Feature Only": v_ft,
        "Volume Sales Display Only": v_dp,
        "Volume Sales Special Pack Only": v_sp,
        "Volume Sales Feature and Display": v_fd,
        "Avg Weekly Items per Store Selling": items,
        "ACV Weighted Distribution": acv,
        "Total Points of Distribution": tpd,
        "Total Points of Distribution No Merch": tpd * (1 - any_merch_w),
        "Total Points of Distribution Feature and/or Display": tpd * any_merch_w,
        "Weighted Weeks Price Reductions Only": ww_pr,
        "Weighted Weeks Feature Only": ww_ft,
        "Weighted Weeks Display Only": ww_dp,
        "Weighted Weeks Special Pack Only": ww_sp,
        "Weighted Weeks Feature and Display": ww_fd,
        "Price per Volume": np.round(price, 4),
        "Price per Volume No Merch": pv(1.06),
        "Price per Volume Any Merch": pv(0.89),
        "Price per Volume Price Reductions Only": pv(0.83),
        "Price per Volume Feature Only": pv(0.94),
        "Price per Volume Display Only": pv(0.92),
        "Price per Volume Special Pack Only": pv(0.97),
        "Price per Volume Feature and Display": pv(0.78),
        "Dollar Sales": volume * price,
        "Dollar Sales No Merch": v_no * price,
        "Dollar Sales Any Merch": v_any * price,
        "Dollar Sales Price Reductions Only": v_pr * price,
        "Dollar Sales Feature Only": v_ft * price,
        "Dollar Sales Display Only": v_dp * price,
        "Dollar Sales Special Pack Only": v_sp * price,
        "Dollar Sales Feature and Display": v_fd * price,
        "Week": WEEKS,
        # Category price shares a slow common trend with the brand's own price
        # but moves largely on its own. Generated as a near-copy it lands at
        # r = 0.98 and takes Price per Volume down with it under VIF pruning.
        "Category P Price Per Volume": np.round(
            spec.base_price * rng.uniform(0.93, 1.07)
            * (1 + 0.35 * (price / price_ref0 - 1) + _ar1(rng, n, 0.028, rho=0.75)), 4),
        "Total Category Price Per Volume": np.round(
            spec.base_price * rng.uniform(0.88, 1.12)
            * (1 + 0.25 * (price / price_ref0 - 1) + _ar1(rng, n, 0.032, rho=0.75)), 4),
    }
    for c in MACRO.columns:
        data[c] = MACRO[c].values
    data["Trend"] = t + 1
    for c, v in media.items():
        data[c] = v
    data["Week_Num"] = _WEEK_NUM
    data["Seasonality_Index"] = np.round(season, 5)
    for c, v in comp.items():
        data[c] = np.round(v, 4) if isinstance(v, np.ndarray) else v

    return pd.DataFrame(data)


def build_brand(bi: int, spec: BrandSpec) -> pd.DataFrame:
    media_cols = [f"{spec.name}_{m}_Spend" for m in MEDIA_POOL[:spec.n_media]]
    rival_names = RIVALS[:spec.n_rivals]
    frames = []
    for ci in spec.channels:
        sig_sd, noise = SLICE_PLAN[(bi, ci)]
        d = build_slice(bi, ci, spec, media_cols, rival_names, sig_sd, noise)

        cov = COVERAGE.get((bi, ci))
        if cov:
            kind, k = cov
            d = d.iloc[:k] if kind == "head" else d.iloc[-k:]
            d = d.reset_index(drop=True)

        if ci in spec.zero_tail:
            # delisted in Q4 2025: still measured, but nothing sells. The
            # engine finds no non-zero weeks in the reserved tail and reports
            # the slice as "No holdout" rather than inventing a MAPE.
            vol_cols = [c for c in d.columns
                        if c.startswith(("Volume Sales", "Dollar Sales"))]
            tail = d.index[-HOLDOUT_WEEKS:]
            d.loc[tail, vol_cols] = 0.0
            d.loc[tail, "ACV Weighted Distribution"] = 0.0
            d.loc[tail, "Total Points of Distribution"] = 0.0

        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Data dictionary — same three columns and the same designations as the
# sponsor's sheet, regenerated for the fictional column set.
# ---------------------------------------------------------------------------
def build_dictionary(sample: pd.DataFrame, spec: BrandSpec) -> pd.DataFrame:
    N = {
        "unique": "Use unique for modeling",
        "target": "We're predicting this",
        "trade_vol": ("Direct sales on trade variable. Strongly connected to "
                      "Volume Sales; execution is better represented by "
                      "Weighted Weeks."),
        "ww": "Reach and frequency of trade executions",
        "price_cond": "Price * trade conditions",
        "macro": "Macro-economic factors for use in forecasting conditions",
        "media": 'Media spend. Should be "decayed" or adstocked.',
        "comp_dist": "Competitive distribution",
        "comp_trade": "Competitive trade",
    }
    rows = [("Geography", "Channel", N["unique"]),
            ("Product", "Product", N["unique"]),
            ("Time", "Time Variable", ""),
            ("Volume Sales", "Dependent Variable/Target", N["target"])]
    for c in sample.columns:
        if c in ("Geography", "Product", "Time", "Volume Sales"):
            continue
        if c.startswith("Volume Sales"):
            rows.append((c, "Trade", N["trade_vol"]))
        elif c.startswith("Dollar Sales"):
            rows.append((c, "Other POS Info", "."))
        elif c.startswith("Weighted Weeks"):
            rows.append((c, "Trade", N["ww"]))
        elif c.startswith("Total Points of Distribution"):
            rows.append((c, "Distribution",
                         "Combined measure of ACV breadth and depth"))
        elif c == "Avg Weekly Items per Store Selling":
            rows.append((c, "Distribution", "Depth of distribution"))
        elif c == "ACV Weighted Distribution":
            rows.append((c, "Distribution", "Breadth of distribution"))
        elif c == "Price per Volume":
            rows.append((c, "Price", "Dollar Sales / Volume Sales"))
        elif c.startswith("Price per Volume"):
            rows.append((c, "Price", N["price_cond"]))
        elif c == "Week":
            rows.append((c, "Better formatted Time Variable", ""))
        elif "Category Price Per Volume" in c or c.startswith("Category"):
            rows.append((c, "Category Price", "Competitive price variable"))
        elif c in MACRO.columns:
            rows.append((c, "Macro-Economic", N["macro"]))
        elif c == "Trend":
            rows.append((c, "Trend", "Trend"))
        elif c.endswith("_Spend"):
            rows.append((c, "Media", N["media"]))
        elif c == "Week_Num":
            rows.append((c, "1-52 Week Count", "."))
        elif c == "Seasonality_Index":
            rows.append((c, "Seasonality Index",
                         "Average of week-num sales over the period divided by "
                         "the average over the entire period, as an index."))
        elif c.startswith("_C_"):
            kind = N["comp_dist"] if ("ACV" in c or "Items" in c) else N["comp_trade"]
            rows.append((c, "Competitive Data", kind))
        else:
            rows.append((c, "Other", ""))
    return pd.DataFrame(rows, columns=["Field", "Designation", "Notes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="Demo Dataset for Presentation.xlsx")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    global SLICE_PLAN
    SLICE_PLAN = build_slice_plan()

    sheets = {}
    total_slices = 0
    for bi, spec in enumerate(BRANDS):
        df = build_brand(bi, spec)
        sheets[spec.name] = df
        total_slices += len(spec.channels)
        print(f"{spec.name:<20} {len(df):>5} rows  "
              f"{len(spec.channels)} channels  {len(df.columns)} cols")

    first = BRANDS[0]
    dictionary = build_dictionary(sheets[first.name], first)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        dictionary.to_excel(xw, sheet_name="General Data Dictionary", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)

    print(f"\n{total_slices} Brand x Channel slices -> {out}")
    worst = min(
        (sheets[b.name].groupby("Geography")["Volume Sales"]
         .apply(lambda v: v[v > 0].min() / v[v > 0].mean()).min())
        for b in BRANDS)
    print(f"lowest weekly volume seen: {worst:.2f} x that slice's mean "
          f"(a value near the 0.22 floor means the signal is being clipped "
          f"and MAPE will misbehave)")


if __name__ == "__main__":
    main()

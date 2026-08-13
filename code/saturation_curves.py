"""
Saturation curve viewing  (Alex Hathcock, "Last Set of Dashboard Updates", 2026-08-11)
═══════════════════════════════════════════════════════════════════════════════

Two separate deliverables live in this module, and it is worth being clear that
they answer two different questions:

1. `media_transform_series` — the WEEKLY DIAGNOSTIC.
   "Weekly plot of raw media, vs decayed media vs decayed and saturated media…
    include correlations with the target for all 3 measures."
   This is about the *model*: it shows exactly what the engine fed into the
   regression for a media variable, at the weekly grain the model actually saw,
   and whether each successive transform tracks the target any better. It is
   what lets a user answer "why was decay 0.4 chosen?" by looking rather than
   trusting.

   Note the scale warning Alex flagged: the saturated series is bounded in
   [0, 1) by construction (see `cp.hill`) while raw and decayed are in
   impressions/dollars. They cannot share a y-axis; the dashboard puts
   saturated on a secondary axis, and correlation — being scale-free — is the
   number that makes the three comparable.

2. `SatCurve` — the ANNUAL ROI GENERALIZATION.
   A faithful port of Alex's own class (`class SatCurve.txt`), which takes the
   latest year of modeling data and produces the execution-vs-ROI chart: sales
   driven (shaded, left axis) against average and marginal return (right axis),
   with x expressed as a % of current spend. Where the two return lines cross
   is the optimal spend level, returned by `optimalpoint()`.

   Alex's own caveat, kept here because it governs how the output should be
   read: "This is a generalization meant for visualizing a whole year's worth
   of execution in terms of saturation and ROI." It is NOT the weekly model.
   It collapses a year of weekly executions to a single average-weekly
   operating point and asks what the response would look like if the whole
   year were scaled up or down together.

DIFFERENCE FROM THE ENGINE'S OWN HILL, DELIBERATELY PRESERVED
─────────────────────────────────────────────────────────────
`cp.hill` parameterizes the half-saturation point as `midpoint × ref`, where
ref is the max of the adstocked series (training rows only). Alex's class uses

    half_saturation = saturation × (max(raw) − min(raw)) + min(raw)

i.e. the midpoint is a fraction of the observed RANGE of raw executions, offset
by the minimum. When the minimum is 0 — the usual case for media — the two
agree exactly. They differ only for an always-on variable with a non-zero
floor. This port keeps Alex's formula because the whole point of the exercise
is that the graph and `optimalpoint()` match the numbers he gets from his own
code; `midpoint_to_alex_saturation` documents the conversion in the other
direction.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The Hill curve, in Alex's parameterization (Kd in the units of x, not a
# fraction). Same function as cp.hill, different argument convention.
# ---------------------------------------------------------------------------

def hill_curve(x, Kd: float, n: float) -> np.ndarray:
    """H(x) = x^n / (x^n + Kd^n).

    Kd is the half-saturation point IN THE UNITS OF x (Alex's class computes it
    from the observed range before calling), n is the slope. Returns [0, 1).
    """
    x = np.asarray(x, dtype=float)
    Kd = float(Kd)
    if not np.isfinite(Kd) or Kd <= 0:
        return np.zeros_like(x)
    xs = np.power(np.clip(x, 0.0, None), float(n))
    ks = Kd ** float(n)
    out = xs / (xs + ks)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


class SatCurve:
    """Execution-vs-ROI response curve for one media variable over one year.

    PORT NOTE — this mirrors Alex's `class SatCurve` line for line, including
    the details that look incidental but change the answer:

      * `average_nonzero_executions` is an `int()` of the mean over weeks with
        execution > 0 — zero weeks are excluded (a flight that ran 12 weeks out
        of 52 has an average execution of its 12 live weeks, not 12/52 of it),
        and the truncation to int is kept so numbers match his output exactly.
      * the x grid is `linspace(0, 300, 301) / 100`, i.e. 0%–300% of current
        spend in 1-point steps. `optimalpoint()` therefore has 1% resolution,
        which is Alex's intent — it is a planning read-out, not a solver.
      * marginal return is the discrete first difference of revenue over the
        first difference of spend along that grid, not an analytic derivative.
      * the optimal point is `argmin |average − marginal|`, with everything
        below 20% of current spend disqualified (`Diff` forced to 100). At very
        low spend both curves are near-zero and would otherwise produce a
        spurious crossing at the origin.

    Parameters
    ----------
    data_in : DataFrame
        The latest year of modeling data for this slice (52 weeks), containing
        the media column.
    tactic : str
        Feature name. A trailing "_S" (Alex's saturated-column suffix) is
        stripped to find the raw column; the label keeps it.
    spend, price, margin : float
        Annual spend on this tactic, the price per unit of the target, and the
        gross margin as a fraction. Revenue = sales × price × margin, so ROI is
        margin dollars returned per dollar spent.
    target_response : float
        The contribution of this variable over that same year — the model's
        answer for "sales driven at current spend". Anchors the curve: the
        fitted Hill shape is scaled so that at 100% spend it returns exactly
        this number.
    saturation : float
        Half-saturation as a fraction of the observed execution range (γ).
    slope : float
        Hill slope (n).
    """

    def __init__(self, data_in, tactic: str, spend: float, price: float,
                 margin: float, target_response: float, saturation: float,
                 slope: float):
        if str(tactic).endswith("_S"):
            self.tactic_main = str(tactic)[:-2]
            self.tactic_sat = str(tactic)
        else:
            self.tactic_main = str(tactic)
            self.tactic_sat = str(tactic) + "_S"

        self.spend = float(spend)
        self.price = float(price)
        self.margin = float(margin)
        self.saturation = float(saturation)
        self.slope = float(slope)
        self.target_response = float(target_response)

        executions = np.asarray(
            pd.to_numeric(data_in[self.tactic_main], errors="coerce")
            .fillna(0).values, dtype=float)
        live = executions[executions > 0]
        if live.size == 0:
            raise ValueError(
                f"{self.tactic_main}: no weeks with execution > 0 in the year "
                "supplied — there is no operating point to build a curve around")
        average_nonzero_executions = int(live.mean())
        if average_nonzero_executions <= 0:
            # executions smaller than 1 unit (already scaled, e.g. spend in
            # $millions) would truncate to 0 and destroy the x axis; keep the
            # un-truncated mean in that case rather than dividing by zero
            average_nonzero_executions = float(live.mean())
        self.average_nonzero_executions = average_nonzero_executions

        half_saturation_unscaled = (self.saturation
                                    * (executions.max() - executions.min())
                                    + executions.min())

        x_final = np.linspace(0, 300, 301) / 100      # 0% .. 300% of current
        x_scaled = x_final * average_nonzero_executions

        self.Kd = float(half_saturation_unscaled)
        self.Kd_P = float(half_saturation_unscaled / average_nonzero_executions)

        y = hill_curve(x_scaled, self.Kd, self.slope)

        # anchor: the curve must pass through (100% spend, contribution)
        y_at_current = float(y[np.isclose(x_final, 1.0)][0])
        scale = (self.target_response / y_at_current) if y_at_current else 0.0
        self.curve_scale = float(scale)

        rcd = pd.DataFrame({"Ratio": x_final, "Curve_Y": y,
                            "Saturated_Sales": y * scale})
        rcd["Title"] = self.tactic_sat
        rcd["Spend"] = rcd["Ratio"] * self.spend
        rcd["Revenue"] = rcd["Saturated_Sales"] * self.price * self.margin
        with np.errstate(divide="ignore", invalid="ignore"):
            rcd["Average Return"] = rcd["Revenue"] / rcd["Spend"]
            rcd["Marginal Return"] = ((rcd["Revenue"] - rcd["Revenue"].shift(1))
                                      / (rcd["Spend"] - rcd["Spend"].shift(1)))
        rcd = rcd.replace([np.inf, -np.inf], np.nan).fillna(0)

        rcd["Diff"] = abs(rcd["Average Return"] - rcd["Marginal Return"])
        rcd.loc[rcd["Ratio"] < .2, "Diff"] = 100      # disqualify the origin

        self._response_curve_data = rcd
        self._optimal_point_ratio = float(rcd.loc[rcd["Diff"].idxmin()]["Ratio"])

    # ── Alex's public API, unchanged ──────────────────────────────────────
    def optimalpoint(self) -> pd.DataFrame:
        """The row where average and marginal return cross — the spend level
        at which the next dollar stops beating the average dollar. THE number
        Alex asked for."""
        return self._response_curve_data[
            self._response_curve_data["Ratio"] == self._optimal_point_ratio]

    def currentpoint(self) -> pd.DataFrame:
        """The row at 100% of current spend."""
        return self._response_curve_data[self._response_curve_data["Ratio"] == 1]

    def response_curve_data(self) -> pd.DataFrame:
        return self._response_curve_data

    def parameters(self) -> dict:
        return {"saturation": self.saturation, "slope": self.slope}

    # ── additions for the dashboard (Alex's class plotted with matplotlib;
    #    the dashboard is Dash/Plotly, so the numbers are handed out as plain
    #    lists and the figure is assembled on the UI side) ─────────────────
    def to_dict(self) -> dict:
        """Plain-python payload: the four series the chart draws plus the two
        read-outs (current and optimal) and the headline verdict."""
        rcd = self._response_curve_data
        opt = self.optimalpoint().iloc[0]
        cur = self.currentpoint().iloc[0]
        ratio = float(self._optimal_point_ratio)

        # ── WHEN THE "OPTIMAL POINT" IS NOT ACTUALLY A CROSSING ───────────
        # The crossover rule assumes the two return lines meet inside the
        # 0–300% grid. They do for an S-curve (slope > 1): marginal return
        # rises, peaks and falls back through the average. For a CONCAVE curve
        # (slope < 1) the ratio marginal/average is the constant `slope` at
        # every spend level, so the two lines never cross — their gap just
        # narrows monotonically and argmin lands on the last grid point.
        # Reporting "optimal = 300%" there would be a graphing artifact
        # presented as a recommendation, so it is flagged instead. The same
        # applies at the 20% floor.
        at_boundary = ratio >= 2.99 or ratio <= 0.21
        if at_boundary:
            verdict = ("no crossover inside 0–300% of current spend — "
                       "returns still diminishing at the edge of the grid"
                       if ratio >= 2.99 else
                       "crossover falls below the 20% floor")
            direction = "inconclusive"
        elif ratio > 1.05:
            verdict, direction = "under-invested", "increase"
        elif ratio < 0.95:
            verdict, direction = "over-invested", "decrease"
        else:
            verdict, direction = "at optimal", "hold"
        return {
            "variable": self.tactic_main,
            "label": self.tactic_sat,
            "ratio": [float(v) for v in rcd["Ratio"]],
            "sales": [float(v) for v in rcd["Saturated_Sales"]],
            "avg_return": [float(v) for v in rcd["Average Return"]],
            "marginal_return": [float(v) for v in rcd["Marginal Return"]],
            "spend_axis": [float(v) for v in rcd["Spend"]],
            "optimal_ratio": ratio,
            "optimal_spend": float(opt["Spend"]),
            "optimal_sales": float(opt["Saturated_Sales"]),
            "optimal_avg_return": float(opt["Average Return"]),
            "optimal_marginal_return": float(opt["Marginal Return"]),
            "current_spend": float(cur["Spend"]),
            "current_sales": float(cur["Saturated_Sales"]),
            "current_avg_return": float(cur["Average Return"]),
            "current_marginal_return": float(cur["Marginal Return"]),
            "half_saturation": float(self.Kd),
            "half_saturation_pct": float(self.Kd_P),
            "avg_weekly_execution": float(self.average_nonzero_executions),
            "saturation": float(self.saturation),
            "slope": float(self.slope),
            "spend": float(self.spend),
            "price": float(self.price),
            "margin": float(self.margin),
            "target_response": float(self.target_response),
            "verdict": verdict,
            "direction": direction,
            "at_boundary": bool(at_boundary),
        }

    def plots(self):                                    # pragma: no cover
        """Alex's original matplotlib figure, kept so his code path still runs
        outside the dashboard (notebooks, ad-hoc checks)."""
        import matplotlib.pyplot as plt
        rcd = self._response_curve_data
        fig, ax1 = plt.subplots(figsize=(10, 6))
        lns1 = ax1.plot(rcd["Ratio"], rcd["Saturated_Sales"], color="tab:cyan",
                        label="Sales", alpha=.2)
        ax1.set_ylabel("Sales Driven")
        ax1.grid(False)
        ax1.fill_between(rcd["Ratio"], 0, rcd["Saturated_Sales"],
                         color="tab:cyan", alpha=.2)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(rcd["Ratio"], rcd["Average Return"], color="tab:blue",
                        label="Average Return")
        lns3 = ax2.plot(rcd["Ratio"], rcd["Marginal Return"], color="tab:red",
                        label="Marginal Return")
        ax1.set_xlabel("% Spend")
        ax2.set_ylabel("$ ROI")
        lns = lns1 + lns2 + lns3
        ax1.legend(lns, [l.get_label() for l in lns], loc=1, framealpha=1)
        plt.title(self.tactic_sat)
        plt.show()
        print(f"HalfSat: {self.Kd}, Slope: {self.slope}")
        return fig


# ---------------------------------------------------------------------------
# Parameter conversion between the two Hill conventions
# ---------------------------------------------------------------------------

def midpoint_to_alex_saturation(midpoint: float, sat_ref: float,
                                raw: np.ndarray) -> float:
    """Convert the engine's `sat_midpoint` (a fraction of the adstocked max)
    into the `saturation` argument Alex's class expects (a fraction of the raw
    range). Identical whenever min(raw) == 0, which holds for any media that is
    ever dark; the general form is kept so an always-on variable is right too.
    """
    raw = np.asarray(raw, dtype=float)
    lo = float(np.nanmin(raw)) if raw.size else 0.0
    hi = float(np.nanmax(raw)) if raw.size else 0.0
    rng = hi - lo
    if rng <= 0:
        return float(midpoint)
    k_units = float(midpoint) * float(sat_ref or hi)     # half-sat in x units
    return float((k_units - lo) / rng)


# ---------------------------------------------------------------------------
# Weekly transform diagnostic  (raw vs decayed vs decayed+saturated)
# ---------------------------------------------------------------------------

def _corr(a, b) -> Optional[float]:
    """Pearson r, or None when either side is constant (a flat series has no
    correlation — reporting 0 would imply 'no relationship' when the truth is
    'undefined')."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def media_transform_series(df_window: pd.DataFrame, variable: str,
                           decay: Optional[float], scale: float = 1.0,
                           sat_midpoint: Optional[float] = None,
                           sat_slope: Optional[float] = None,
                           sat_ref: Optional[float] = None,
                           target: Optional[pd.Series] = None,
                           dates: Optional[pd.Series] = None) -> dict:
    """Weekly raw / decayed / decayed+saturated series for one media variable,
    with each one's correlation to the target.

    Follows the SAME transform order the model uses — raw → adstock → scale →
    Hill (`cp.assemble_matrix`) — so what is plotted is literally the column
    that entered the regression, not a re-derivation that might drift from it.

    `saturated` is returned on the model's own [0, 1) scale AND rescaled to the
    decayed series' maximum (`saturated_rescaled`) purely so the shape can be
    read against the other two on one chart; correlations are computed on the
    unscaled series (a positive linear rescale cannot change Pearson r, so the
    two agree — the rescale is presentation only).

    Alex asked for the target on the chart as well ("bonus points"), so it is
    returned alongside, likewise rescaled for shape comparison.
    """
    import capstone_pipeline as cp

    raw = np.asarray(pd.to_numeric(df_window[variable], errors="coerce")
                     .fillna(0).values, dtype=float)
    decayed = cp.adstock(raw, decay) if decay is not None else raw.copy()
    sc = float(scale or 1.0)
    decayed_scaled = decayed / sc if sc != 1.0 else decayed

    saturated = None
    if sat_midpoint is not None and sat_slope is not None:
        ref = sat_ref if sat_ref else (float(np.nanmax(decayed_scaled))
                                       if decayed_scaled.size else 0.0)
        saturated = cp.hill(decayed_scaled, sat_midpoint, sat_slope, ref=ref)

    tgt = (np.asarray(pd.to_numeric(target, errors="coerce").fillna(0).values,
                      dtype=float) if target is not None else None)

    def _rescale(series, to):
        """Linear rescale to the peak of `to`, for shape comparison only."""
        if series is None:
            return None
        s = np.asarray(series, dtype=float)
        smax = float(np.nanmax(s)) if s.size else 0.0
        tmax = float(np.nanmax(to)) if len(to) else 0.0
        if smax <= 0 or tmax <= 0:
            return [float(v) for v in s]
        return [float(v) for v in s * (tmax / smax)]

    out = {
        "variable": variable,
        "dates": ([d.strftime("%Y-%m-%d") for d in pd.to_datetime(dates)]
                  if dates is not None else list(range(len(raw)))),
        "raw": [float(v) for v in raw],
        "decayed": [float(v) for v in decayed],
        "saturated": (None if saturated is None
                      else [float(v) for v in saturated]),
        "saturated_rescaled": _rescale(saturated, decayed),
        "target": (None if tgt is None else [float(v) for v in tgt]),
        "target_rescaled": _rescale(tgt, raw) if tgt is not None else None,
        "decay": (None if decay is None else float(decay)),
        "scale": sc,
        "sat_midpoint": sat_midpoint, "sat_slope": sat_slope,
        "weeks_live": int((raw > 0).sum()),
        "corr": {},
    }
    if tgt is not None:
        out["corr"] = {
            "raw": _corr(raw, tgt),
            "decayed": _corr(decayed, tgt),
            "saturated": (None if saturated is None else _corr(saturated, tgt)),
        }
        # which transform tracks the target best — the answer to "why was this
        # decay/midpoint picked", stated rather than left to the eye
        ranked = [(k, abs(v)) for k, v in out["corr"].items() if v is not None]
        out["best_transform"] = (max(ranked, key=lambda kv: kv[1])[0]
                                 if ranked else None)
    return out


def latest_year(df_window: pd.DataFrame, weeks: int = 52,
                date_col: str = "date") -> pd.DataFrame:
    """The most recent `weeks` rows of a modeling window — the 'latest year of
    the modeling data' Alex's class expects as `data_in`."""
    d = df_window.sort_values(date_col) if date_col in df_window.columns \
        else df_window
    return d.tail(int(weeks)).reset_index(drop=True)


def curve_inputs_from_result(r: dict, variable: str, weeks: int = 52,
                             margin: float = 0.30,
                             price: Optional[float] = None,
                             spend: Optional[float] = None) -> dict:
    """Auto-derive the four inputs Alex's class needs from a `cp.run_slice`
    result, so the user is never asked to type a number the data already knows.
    Every one of them is overridable in the UI.

      spend           Σ raw execution over the latest year. In this dataset the
                      media columns ARE spend (`Brand 1_Amazon_Spend`), so the
                      sum is the annual spend. If a client ever supplies
                      impressions instead, this is the number to override.
      price           mean `Price per Volume` over the same weeks — revenue per
                      unit of the target (Volume Sales).
      margin          NOT derivable from the data. Defaults to 30% and is a
                      user input; ROI scales linearly with it, so it moves the
                      return axis but NOT the optimal point (both average and
                      marginal return scale together, so the crossing is
                      unchanged) — worth knowing before arguing about it.
      target_response Σ of the model's contribution for this variable over the
                      same weeks: sales the model says this media drove.
    """
    df = r["df"]
    n = min(int(weeks), len(df))
    idx = slice(len(df) - n, len(df))
    dfy = df.iloc[idx].reset_index(drop=True)

    raw = pd.to_numeric(dfy[variable], errors="coerce").fillna(0)
    if spend is None:
        spend = float(raw.sum())

    if price is None:
        for c in ("Price per Volume", "Price per Unit", "Price"):
            if c in dfy.columns:
                v = pd.to_numeric(dfy[c], errors="coerce")
                v = v[v > 0]
                if len(v):
                    price = float(v.mean())
                    break
    if price is None or not np.isfinite(price) or price <= 0:
        price = 1.0

    contrib = r["fit"].contributions
    target_response = (float(contrib[variable].values[idx].sum())
                       if variable in contrib.columns else 0.0)

    spec = r["specs_by_name"].get(variable)
    mid = getattr(spec, "sat_midpoint", None) if spec else None
    slp = getattr(spec, "sat_slope", None) if spec else None
    ref = (r.get("sat_refs") or {}).get(variable)
    if mid is not None and slp is not None:
        saturation = midpoint_to_alex_saturation(mid, ref, raw.values)
        slope = float(slp)
    else:
        # The variable was modeled LINEAR — there is no fitted saturation to
        # draw. Rather than refuse the chart, fall back to the midpoint of the
        # search grid so the user sees the shape their spend WOULD imply, and
        # flag it so nobody reads a fitted result into an assumption.
        saturation, slope = 0.5, 1.0
    return {
        "data_in": dfy, "tactic": variable, "spend": spend, "price": price,
        "margin": float(margin), "target_response": target_response,
        "saturation": float(np.clip(saturation, 0.01, 0.99)),
        "slope": float(slope),
        "assumed_curve": mid is None or slp is None,
        "weeks": n,
    }


def build_sat_curve(r: dict, variable: str, weeks: int = 52,
                    margin: float = 0.30, price: Optional[float] = None,
                    spend: Optional[float] = None,
                    saturation: Optional[float] = None,
                    slope: Optional[float] = None) -> dict:
    """`curve_inputs_from_result` → `SatCurve` → payload, with any input
    overridden by the user honoured. Returns `{"error": ...}` rather than
    raising, so one dead variable can't take the whole screen down."""
    try:
        kw = curve_inputs_from_result(r, variable, weeks=weeks, margin=margin,
                                      price=price, spend=spend)
        if saturation is not None:
            kw["saturation"] = float(saturation)
        if slope is not None:
            kw["slope"] = float(slope)
        assumed = kw.pop("assumed_curve")
        n_weeks = kw.pop("weeks")
        curve = SatCurve(**kw)
        out = curve.to_dict()
        out["assumed_curve"] = bool(assumed)
        out["weeks"] = int(n_weeks)
        return out
    except Exception as exc:                       # noqa: BLE001 — surfaced in UI
        return {"variable": variable, "error": str(exc)}

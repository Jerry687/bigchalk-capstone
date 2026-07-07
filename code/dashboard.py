"""
Phase 4: Plotly Dash dashboard for the Big Chalk regression engine.

Load ANY client datafile from the UI (per project scope: "an input of a
datafile with all products*channels and variables"), then:

  1. Run & diagnostics  — pick Product x Channel x target, run, inspect fit.
  2. Variable controls  — per-product editable config: family, sign, custom
     coefficient bounds, adstock decay, role (auto / force / exclude).
     Saved per dataset+product under ../configs/; every run reads it.
  3. Contributions      — due-tos by model year with YoY change, and average
     weekly contribution per driver (media over execution weeks only).
  4. Batch overview     — all-slices summary from run_all.py; click a row to
     load that slice into the controls, then hit Run model.

Product sheets are auto-detected: sheets named "Brand *" if present,
otherwise any sheet containing both `Geography` and `Time` columns.
Target choices come from the loaded data, not from code.

    pip install dash
    cd code && python dashboard.py     # http://127.0.0.1:8050
"""
import os
from functools import lru_cache

# pin BLAS threads before numpy loads (see run_all.py)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State

import capstone_pipeline as cp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
CFG_DIR = os.path.join(ROOT, "configs")
SUMMARY = os.path.join(ROOT, "outputs", "all_models_summary.csv")
os.makedirs(CFG_DIR, exist_ok=True)

WINDOW_CHOICES = [52, 104, 156]
CFG_COLS = ["variable", "family", "sign", "adstock_decay",
            "coef_lower", "coef_upper", "role"]
PREFERRED_TARGETS = ["Volume Sales", "Dollar Sales", "Unit Sales"]


def _slug(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


@lru_cache(maxsize=24)
def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


@lru_cache(maxsize=4)
def detect_product_sheets(path: str) -> tuple:
    """Sheets named 'Brand *' if any; otherwise any sheet that has both
    Geography and Time columns (handles arbitrary client naming)."""
    xl = pd.ExcelFile(path)
    named = tuple(s for s in xl.sheet_names if s.lower().startswith("brand"))
    if named:
        return named
    found = []
    for s in xl.sheet_names:
        try:
            head = pd.read_excel(xl, sheet_name=s, nrows=1)
        except Exception:
            continue
        if {"Geography", "Time"} <= set(map(str, head.columns)):
            found.append(s)
    return tuple(found)


def channels_for(path: str, sheet: str) -> list:
    g = load_sheet(path, sheet)["Geography"].unique()
    return [c for c in g if isinstance(c, str)]


def target_choices(path: str, sheet: str) -> list:
    """Candidate dependent variables from the data itself: preferred sales
    measures first, then any other numeric non-competitive column."""
    df = load_sheet(path, sheet)
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c])
           and not str(c).startswith("_C_")]
    pref = [c for c in PREFERRED_TARGETS if c in df.columns]
    rest = sorted(c for c in num if c not in pref)
    return pref + rest


def cfg_path(path: str, sheet: str) -> str:
    ds = _slug(os.path.splitext(os.path.basename(path))[0])[:24]
    return os.path.join(CFG_DIR, f"varconfig_{ds}_{_slug(sheet)}.csv")


def batch_summary_path(path: str) -> str:
    ds = _slug(os.path.splitext(os.path.basename(path))[0])[:24]
    return os.path.join(ROOT, "outputs", f"batch_summary_{ds}.csv")


def run_batch(path: str, target_pref: str, window: int) -> pd.DataFrame:
    """Run every Product x Channel of a datafile, using each product's saved
    variable config (the same one edited in the Variable controls tab)."""
    rows = []
    for sheet in detect_product_sheets(path):
        df_sheet = load_sheet(path, sheet)
        load_or_create_config(path, sheet)
        tgt = target_pref if target_pref in df_sheet.columns else next(
            (t for t in PREFERRED_TARGETS if t in df_sheet.columns), None)
        if tgt is None:
            continue
        for channel in channels_for(path, sheet):
            if df_sheet.loc[df_sheet["Geography"] == channel, tgt].abs().sum() == 0:
                continue                      # not sold in this channel
            try:
                cfg = cp.ModelConfig(target=tgt, model_weeks=int(window),
                                     variable_config=cfg_path(path, sheet))
                r = cp.run_slice("", sheet, channel, config=cfg, df=df_sheet)
                fit = r["fit"]
                rows.append({
                    "brand": sheet, "channel": channel, "target": tgt,
                    "n_selected": len(r["selected"]),
                    "n_forced": len(r["forced"]),
                    "R2": round(fit.r2, 4),
                    "MAPE_in_pct": round(fit.mape, 2),
                    "MAPE_holdout_pct": round(r["holdout_mape"], 2),
                    "n_sign_conflicts": len(r["sign_conflicts"]),
                })
            except Exception as e:
                rows.append({"brand": sheet, "channel": channel, "target": tgt,
                             "n_selected": 0, "n_forced": 0, "R2": np.nan,
                             "MAPE_in_pct": np.nan, "MAPE_holdout_pct": np.nan,
                             "n_sign_conflicts": 0, "error": str(e)})
    out = pd.DataFrame(rows).sort_values("MAPE_holdout_pct")
    out.to_csv(batch_summary_path(path), index=False)
    return out


def load_or_create_config(path: str, sheet: str,
                          regenerate: bool = False) -> pd.DataFrame:
    """Per-dataset, per-product variable config; generated from heuristics on
    first use. ACV Weighted Distribution defaults to role=force (validated:
    a volume model without a distribution term cannot track shelf change)."""
    p = cfg_path(path, sheet)
    if os.path.exists(p) and not regenerate:
        return pd.read_csv(p)
    cfg = cp.generate_variable_config(load_sheet(path, sheet))
    cfg.loc[cfg["variable"] == "ACV Weighted Distribution", "role"] = "force"
    cfg.to_csv(p, index=False)
    return cfg


app = Dash(__name__, title="Big Chalk MMM Engine")

CARD = {"padding": "12px 18px", "borderRadius": "8px", "background": "#f7fafc",
        "border": "1px solid #e2e8f0", "textAlign": "center", "minWidth": "120px"}
BTN = {"height": "38px", "padding": "0 24px", "border": "none",
       "borderRadius": "6px", "cursor": "pointer"}
TBL_STYLE = dict(style_cell={"fontFamily": "Segoe UI", "fontSize": "13px",
                             "textAlign": "left"},
                 style_header={"fontWeight": "600"})

app.layout = html.Div(style={"fontFamily": "Segoe UI, sans-serif",
                             "margin": "0 auto", "maxWidth": "1200px",
                             "padding": "16px"}, children=[
    html.H2("Big Chalk — Marketing-Mix Regression Engine"),
    dcc.Store(id="datapath"),
    dcc.Store(id="batch_sel"),

    html.Div(style={"display": "flex", "gap": "12px", "alignItems": "flex-end",
                    "flexWrap": "wrap", "marginBottom": "10px"}, children=[
        html.Div(style={"flex": 1, "minWidth": "380px"}, children=[
            html.Label("Data file (.xlsx — one sheet per product, rows = "
                       "channel × week)"),
            dcc.Input(id="datafile_input", type="text",
                      value=DEFAULT_DATA if os.path.exists(DEFAULT_DATA) else "",
                      style={"width": "100%", "height": "34px",
                             "padding": "0 8px"}),
        ]),
        html.Button("Load data", id="load_data", n_clicks=0,
                    style={**BTN, "background": "#2b6cb0", "color": "white"}),
        html.Div(id="load_status", style={"color": "#4a5568",
                                          "fontSize": "13px",
                                          "maxWidth": "360px"}),
    ]),

    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                    "alignItems": "flex-end"}, children=[
        html.Div([html.Label("Product"),
                  dcc.Dropdown(id="brand", clearable=False,
                               style={"width": "180px"})]),
        html.Div([html.Label("Channel"),
                  dcc.Dropdown(id="channel", clearable=False,
                               style={"width": "160px"})]),
        html.Div([html.Label("Target"),
                  dcc.Dropdown(id="target", clearable=False,
                               style={"width": "260px"})]),
        html.Div([html.Label("Model window (weeks)"),
                  dcc.Dropdown(id="window", options=WINDOW_CHOICES, value=104,
                               clearable=False, style={"width": "150px"})]),
        html.Button("Run model", id="run", n_clicks=0,
                    style={**BTN, "background": "#4c51bf", "color": "white"}),
    ]),

    dcc.Tabs(style={"marginTop": "14px"}, children=[

        dcc.Tab(label="Run & diagnostics", children=[dcc.Loading(children=[
            html.Div(id="metrics", style={"display": "flex", "gap": "12px",
                                          "margin": "18px 0",
                                          "flexWrap": "wrap"}),
            dcc.Graph(id="fig_fit"),
            html.Div(style={"display": "flex", "gap": "12px"}, children=[
                dcc.Graph(id="fig_resid", style={"flex": 1}),
                dcc.Graph(id="fig_residhist", style={"flex": 1}),
            ]),
            html.H4("Coefficients"),
            dash_table.DataTable(id="coef_table", **TBL_STYLE),
            html.Div(id="err", style={"color": "#c53030",
                                      "marginTop": "8px"}),
        ])]),

        dcc.Tab(label="Variable controls", children=[
            html.Div(style={"display": "flex", "gap": "12px",
                            "margin": "14px 0", "alignItems": "center"},
                     children=[
                html.Button("Save config", id="cfg_save", n_clicks=0,
                            style={**BTN, "background": "#38a169",
                                   "color": "white"}),
                html.Button("Regenerate defaults", id="cfg_regen", n_clicks=0,
                            style={**BTN, "background": "#e2e8f0"}),
                html.Div(id="cfg_status", style={"color": "#4a5568"}),
            ]),
            html.Div("Edit cells directly. sign: positive / negative / "
                     "unconstrained · role: auto (selection decides) / force "
                     "(always in — client-mandated) / exclude (never in) · "
                     "coef bounds override the sign · adstock_decay only "
                     "applies to media.",
                     style={"fontSize": "13px", "color": "#4a5568",
                            "marginBottom": "10px"}),
            dash_table.DataTable(
                id="cfg_table", editable=True, page_size=200,
                columns=[
                    {"id": "variable", "name": "variable", "editable": False},
                    {"id": "family", "name": "family"},
                    {"id": "sign", "name": "sign",
                     "presentation": "dropdown"},
                    {"id": "adstock_decay", "name": "adstock_decay",
                     "type": "numeric"},
                    {"id": "coef_lower", "name": "coef_lower",
                     "type": "numeric"},
                    {"id": "coef_upper", "name": "coef_upper",
                     "type": "numeric"},
                    {"id": "role", "name": "role",
                     "presentation": "dropdown"},
                ],
                dropdown={
                    "sign": {"options": [{"label": s, "value": s} for s in
                             ["positive", "negative", "unconstrained"]]},
                    "role": {"options": [{"label": s, "value": s} for s in
                             ["auto", "force", "exclude"]]},
                },
                style_cell={"fontFamily": "Segoe UI", "fontSize": "13px",
                            "textAlign": "left", "minWidth": "90px"},
                style_header={"fontWeight": "600"},
                style_data_conditional=[
                    {"if": {"filter_query": '{role} = "force"'},
                     "backgroundColor": "#ebf8ff"},
                    {"if": {"filter_query": '{role} = "exclude"'},
                     "backgroundColor": "#fff5f5"},
                ],
            ),
        ]),

        dcc.Tab(label="Contributions", children=[dcc.Loading(children=[
            html.Div("Run a model first (any tab) — contributions update with "
                     "every run. Due-tos decompose the fitted line exactly: "
                     "intercept + sum of coefficient × value per week.",
                     style={"fontSize": "13px", "color": "#4a5568",
                            "margin": "12px 0"}),
            dcc.Graph(id="fig_cby"),
            dcc.Graph(id="fig_avgc"),
            html.H4("Due-tos by model year"),
            dash_table.DataTable(id="yoy_table", **TBL_STYLE),
        ])]),

        dcc.Tab(label="Batch overview", children=[
            html.Div(style={"display": "flex", "gap": "12px",
                            "margin": "14px 0", "alignItems": "center"},
                     children=[
                html.Button("Run all combinations", id="batch_run",
                            n_clicks=0,
                            style={**BTN, "background": "#4c51bf",
                                   "color": "white"}),
                html.Button("Reload summary", id="batch_refresh", n_clicks=0,
                            style={**BTN, "background": "#e2e8f0"}),
                html.Div(id="batch_status", style={"color": "#4a5568"}),
            ]),
            html.Div("Run all combinations batches every Product × "
                     "Channel of the LOADED datafile with the saved variable "
                     "configs (target/window from the controls above). Sort "
                     "by any column; click a row to load that slice, then "
                     "hit Run model. Red rows: holdout MAPE > 40% — "
                     "structural-change review candidates.",
                     style={"fontSize": "13px", "color": "#4a5568",
                            "marginBottom": "10px"}),
            dash_table.DataTable(
                id="batch_table", sort_action="native",
                filter_action="native", row_selectable="single",
                page_size=100,
                style_cell={"fontFamily": "Segoe UI", "fontSize": "13px",
                            "textAlign": "left"},
                style_header={"fontWeight": "600"},
                style_data_conditional=[
                    {"if": {"filter_query": "{MAPE_holdout_pct} > 40"},
                     "backgroundColor": "#fff5f5"},
                ],
            ),
        ]),
    ]),
])


# ---------------- callbacks ----------------

@app.callback(Output("datapath", "data"), Output("load_status", "children"),
              Input("load_data", "n_clicks"),
              State("datafile_input", "value"))
def _load_data(_n, path):
    path = (path or "").strip().strip('"')
    if not path:
        return None, "Enter a path to an .xlsx datafile."
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        sheets = detect_product_sheets(path)
    except Exception as e:
        return None, f"Could not read workbook: {e}"
    if not sheets:
        return None, ("No product sheets found (need `Geography` and `Time` "
                      "columns, or sheets named 'Brand *').")
    n_ch = len(channels_for(path, sheets[0]))
    n_wk = len(load_sheet(path, sheets[0])) // max(n_ch, 1)
    return ({"path": path, "sheets": list(sheets)},
            f"Loaded {os.path.basename(path)}: {len(sheets)} product sheets, "
            f"~{n_ch} channels, ~{n_wk} weeks each.")


@app.callback(Output("brand", "options"), Output("brand", "value"),
              Input("datapath", "data"))
def _brands(dp):
    if not dp:
        return [], None
    return dp["sheets"], dp["sheets"][0]


@app.callback(Output("channel", "options"), Output("channel", "value"),
              Input("brand", "value"), Input("batch_sel", "data"),
              State("datapath", "data"))
def _channels(brand, sel, dp):
    if not dp or not brand:
        return [], None
    ch = channels_for(dp["path"], brand)
    if sel and sel.get("brand") == brand and sel.get("channel") in ch:
        return ch, sel["channel"]
    return ch, (ch[0] if ch else None)


@app.callback(Output("target", "options"), Output("target", "value"),
              Input("brand", "value"), State("datapath", "data"),
              State("target", "value"))
def _targets(brand, dp, current):
    if not dp or not brand:
        return [], None
    opts = target_choices(dp["path"], brand)
    if current in opts:
        return opts, current
    return opts, (opts[0] if opts else None)


@app.callback(Output("cfg_table", "data"),
              Input("brand", "value"), Input("cfg_regen", "n_clicks"),
              State("datapath", "data"))
def _load_cfg(brand, _regen, dp):
    if not dp or not brand:
        return []
    try:
        from dash import ctx
        regen = ctx.triggered_id == "cfg_regen"
    except Exception:          # outside a live callback (tests)
        regen = False
    cfg = load_or_create_config(dp["path"], brand, regenerate=regen)
    return cfg[CFG_COLS].to_dict("records")


@app.callback(Output("cfg_status", "children"),
              Input("cfg_save", "n_clicks"),
              State("cfg_table", "data"), State("brand", "value"),
              State("datapath", "data"), prevent_initial_call=True)
def _save_cfg(_n, rows, brand, dp):
    if not dp or not brand:
        return "Load a datafile first."
    cfg = pd.DataFrame(rows, columns=CFG_COLS)
    for c in ("adstock_decay", "coef_lower", "coef_upper"):
        cfg[c] = pd.to_numeric(cfg[c], errors="coerce")
    bad = cfg[(cfg.coef_lower.notna()) & (cfg.coef_upper.notna())
              & (cfg.coef_lower > cfg.coef_upper)]
    if len(bad):
        return f"NOT saved: coef_lower > coef_upper for {list(bad.variable)}"
    p = cfg_path(dp["path"], brand)
    cfg.to_csv(p, index=False)
    n_force = int((cfg.role == "force").sum())
    n_excl = int((cfg.role == "exclude").sum())
    return (f"Saved {os.path.basename(p)} — {len(cfg)} variables, "
            f"{n_force} forced, {n_excl} excluded. Next run uses it.")


def _summary_for(dp):
    """Per-dataset batch summary; falls back to the run_all.py output for
    the bundled dataset."""
    if dp and os.path.exists(batch_summary_path(dp["path"])):
        return batch_summary_path(dp["path"])
    if dp and os.path.normpath(dp["path"]) == os.path.normpath(DEFAULT_DATA) \
            and os.path.exists(SUMMARY):
        return SUMMARY
    return None


def _batch_payload(s):
    cols = [{"name": c, "id": c} for c in s.columns]
    return (s.round(3).to_dict("records"), cols,
            f"{len(s)} models · median R² {s.R2.median():.2f} · "
            f"median holdout MAPE {s.MAPE_holdout_pct.median():.1f}%")


@app.callback(Output("batch_table", "data"), Output("batch_table", "columns"),
              Output("batch_status", "children"),
              Input("batch_refresh", "n_clicks"), Input("datapath", "data"))
def _load_batch(_n, dp):
    p = _summary_for(dp)
    if p is None:
        return [], [], "No batch yet — hit Run all combinations."
    return _batch_payload(pd.read_csv(p))


@app.callback(Output("batch_table", "data", allow_duplicate=True),
              Output("batch_table", "columns", allow_duplicate=True),
              Output("batch_status", "children", allow_duplicate=True),
              Input("batch_run", "n_clicks"),
              State("datapath", "data"), State("target", "value"),
              State("window", "value"), prevent_initial_call=True)
def _run_batch(_n, dp, target, window):
    if not dp:
        return [], [], "Load a datafile first."
    import time as _time
    t0 = _time.time()
    s = run_batch(dp["path"], target or "Volume Sales", window or 104)
    data, cols, status = _batch_payload(s)
    return data, cols, f"{status} · ran in {_time.time()-t0:.0f}s"


@app.callback(Output("brand", "value", allow_duplicate=True),
              Output("batch_sel", "data"),
              Input("batch_table", "selected_rows"),
              State("batch_table", "data"), prevent_initial_call=True)
def _pick_slice(sel, data):
    if not sel or not data:
        from dash import no_update
        return no_update, no_update
    row = data[sel[0]]
    return row["brand"], {"brand": row["brand"], "channel": row["channel"]}


def _metric(label, value):
    return html.Div(style=CARD, children=[
        html.Div(label, style={"fontSize": "12px", "color": "#4a5568"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "600"}),
    ])


@app.callback(
    Output("metrics", "children"), Output("fig_fit", "figure"),
    Output("fig_resid", "figure"), Output("fig_residhist", "figure"),
    Output("coef_table", "data"), Output("coef_table", "columns"),
    Output("fig_cby", "figure"), Output("fig_avgc", "figure"),
    Output("yoy_table", "data"), Output("yoy_table", "columns"),
    Output("err", "children"),
    Input("run", "n_clicks"),
    State("brand", "value"), State("channel", "value"),
    State("target", "value"), State("window", "value"),
    State("datapath", "data"),
    prevent_initial_call=False)
def run_model(_, brand, channel, target, window, dp):
    empty = go.Figure()
    blank = ([], empty, empty, empty, [], [], empty, empty, [], [])
    if not dp or not channel or not target:
        return (*blank, "")
    path = dp["path"]
    try:
        load_or_create_config(path, brand)    # ensure config exists
        cfg = cp.ModelConfig(target=target, model_weeks=int(window),
                             variable_config=cfg_path(path, brand))
        r = cp.run_slice("", brand, channel, config=cfg,
                         df=load_sheet(path, brand))
    except Exception as e:
        return (*blank, f"Model failed: {e}")

    fit, df, split = r["fit"], r["df"], r["split"]
    y = r["y"]

    metrics = [
        _metric("R²", f"{fit.r2:.3f}"),
        _metric("Adj R²", f"{fit.adj_r2:.3f}"),
        _metric("In-sample MAPE", f"{fit.mape:.1f}%"),
        _metric("Holdout MAPE", "n/a" if np.isnan(r["holdout_mape"])
                else f"{r['holdout_mape']:.1f}%"),
        _metric("Durbin-Watson", f"{fit.meta['durbin_watson']:.2f}"),
        _metric("Predictors", f"{len(r['selected'])}"),
    ]

    f1 = go.Figure()
    f1.add_scatter(x=df["date"], y=y, name="Actual",
                   line=dict(color="#2d3748", width=2))
    f1.add_scatter(x=df["date"].iloc[:split], y=fit.fitted, name="Fitted",
                   line=dict(color="#e53e3e", width=2, dash="dash"))
    if len(r["pred_te"]):
        f1.add_scatter(x=df["date"].iloc[split:], y=r["pred_te"],
                       name="Holdout forecast",
                       line=dict(color="#d69e2e", width=2, dash="dash"))
        f1.add_vline(x=df["date"].iloc[split], line_dash="dot",
                     line_color="grey")
    f1.update_layout(title=f"Actual vs Fitted — {brand} × {channel} ({target})",
                     margin=dict(t=48, b=24), height=380,
                     legend=dict(orientation="h"))

    f2 = go.Figure(go.Scatter(x=fit.fitted, y=fit.resid, mode="markers",
                              marker=dict(color="#3182ce", opacity=0.6)))
    f2.add_hline(y=0, line_dash="dash", line_color="red")
    f2.update_layout(title="Residuals vs Fitted", height=320,
                     margin=dict(t=48, b=24))

    f3 = go.Figure(go.Histogram(x=fit.resid, marker_color="#805ad5"))
    f3.update_layout(title="Residual distribution", height=320,
                     margin=dict(t=48, b=24))

    coef_rows = []
    for name in ["const"] + r["selected"]:
        spec = r["specs_by_name"].get(name)
        coef_rows.append({
            "variable": name,
            "family": spec.family if spec else "(intercept)",
            "sign": spec.sign if spec else "",
            "forced": "yes" if name in r["forced"] else "",
            "coefficient": round(float(fit.coef[name]), 4),
            "t (OLS ref)": round(float(fit.tstats.get(name, np.nan)), 2)
                           if name != "const" else None,
        })
    cols = [{"name": k, "id": k} for k in coef_rows[0].keys()]

    # ---- contributions tab ----
    cby = r["contrib_by_year"]
    year_cols = [c for c in cby.columns if not str(c).startswith("YoY")]
    plot_tbl = cby[year_cols].drop("Intercept", errors="ignore")
    f4 = go.Figure()
    palette = ["#a0aec0", "#4c51bf", "#38a169"]
    for i, yc in enumerate(year_cols):
        f4.add_bar(y=plot_tbl.index, x=plot_tbl[yc], name=str(yc),
                   orientation="h",
                   marker_color=palette[i % len(palette)])
    f4.update_layout(barmode="group", height=max(360, 34 * len(plot_tbl)),
                     title=f"Due-to totals by model year — {brand} × {channel}",
                     margin=dict(t=48, b=24), legend=dict(orientation="h"))

    avg = (r["avg_contrib"]["avg_weekly_contribution"]
           .drop("Intercept", errors="ignore").sort_values())
    f5 = go.Figure(go.Bar(
        y=avg.index, x=avg.values, orientation="h",
        marker_color=["#e53e3e" if v < 0 else "#4c51bf" for v in avg.values]))
    f5.update_layout(height=max(360, 34 * len(avg)),
                     title="Avg weekly due-to per driver (signed; media "
                           "averaged over execution weeks only)",
                     margin=dict(t=48, b=24))

    yoy = cby.round(0).reset_index().rename(columns={"index": "driver"})
    yoy.columns = [str(c) for c in yoy.columns]
    yoy_cols = [{"name": c, "id": c} for c in yoy.columns]

    return (metrics, f1, f2, f3, coef_rows, cols,
            f4, f5, yoy.to_dict("records"), yoy_cols, "")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

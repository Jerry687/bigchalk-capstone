"""
Phase 4: Plotly Dash dashboard for the Big Chalk regression engine.

Tab 1 — Run & diagnostics: pick Brand x Channel x target, set the modeling
window, run the constrained model, inspect fit quality.

Tab 2 — Variable controls (Alex's key ask): per-brand editable table of
every candidate variable — family, expected sign, custom coefficient
bounds, adstock decay, and role (auto / force / exclude). Saved to
../configs/variable_config_<brand>.csv; every model run reads the selected
brand's config, so edits here directly steer the engine.

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

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
CFG_DIR = os.path.join(ROOT, "configs")
os.makedirs(CFG_DIR, exist_ok=True)

TARGET_CHOICES = ["Volume Sales", "Dollar Sales"]
WINDOW_CHOICES = [52, 104, 156]
CFG_COLS = ["variable", "family", "sign", "adstock_decay",
            "coef_lower", "coef_upper", "role"]


@lru_cache(maxsize=12)
def load_sheet(brand: str) -> pd.DataFrame:
    return pd.read_excel(DATA, sheet_name=brand)


@lru_cache(maxsize=1)
def brand_names() -> tuple:
    xl = pd.ExcelFile(DATA)
    return tuple(s for s in xl.sheet_names if s.lower().startswith("brand"))


def channels_for(brand: str) -> list:
    g = load_sheet(brand)["Geography"].unique()
    return [c for c in g if isinstance(c, str)]


def cfg_path(brand: str) -> str:
    return os.path.join(CFG_DIR, f"variable_config_{brand.replace(' ', '').lower()}.csv")


def load_or_create_config(brand: str, regenerate: bool = False) -> pd.DataFrame:
    """Per-brand variable config; generated from heuristics on first use.
    ACV Weighted Distribution defaults to role=force (validated: a volume
    model without a distribution term cannot track shelf-presence change)."""
    path = cfg_path(brand)
    if os.path.exists(path) and not regenerate:
        return pd.read_csv(path)
    cfg = cp.generate_variable_config(load_sheet(brand))
    cfg.loc[cfg["variable"] == "ACV Weighted Distribution", "role"] = "force"
    cfg.to_csv(path, index=False)
    return cfg


app = Dash(__name__, title="Big Chalk MMM Engine")

CARD = {"padding": "12px 18px", "borderRadius": "8px", "background": "#f7fafc",
        "border": "1px solid #e2e8f0", "textAlign": "center", "minWidth": "120px"}
BTN = {"height": "38px", "padding": "0 24px", "border": "none",
       "borderRadius": "6px", "cursor": "pointer"}

app.layout = html.Div(style={"fontFamily": "Segoe UI, sans-serif",
                             "margin": "0 auto", "maxWidth": "1200px",
                             "padding": "16px"}, children=[
    html.H2("Big Chalk — Marketing-Mix Regression Engine"),
    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                    "alignItems": "flex-end"}, children=[
        html.Div([html.Label("Brand"),
                  dcc.Dropdown(id="brand", options=list(brand_names()),
                               value="Brand 1", clearable=False,
                               style={"width": "160px"})]),
        html.Div([html.Label("Channel"),
                  dcc.Dropdown(id="channel", clearable=False,
                               style={"width": "160px"})]),
        html.Div([html.Label("Target"),
                  dcc.Dropdown(id="target", options=TARGET_CHOICES,
                               value="Volume Sales", clearable=False,
                               style={"width": "180px"})]),
        html.Div([html.Label("Model window (weeks)"),
                  dcc.Dropdown(id="window", options=WINDOW_CHOICES, value=104,
                               clearable=False, style={"width": "160px"})]),
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
            dash_table.DataTable(id="coef_table",
                                 style_cell={"fontFamily": "Segoe UI",
                                             "fontSize": "13px",
                                             "textAlign": "left"},
                                 style_header={"fontWeight": "600"}),
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
    ]),
])


# ---------------- callbacks ----------------

@app.callback(Output("channel", "options"), Output("channel", "value"),
              Input("brand", "value"))
def _channels(brand):
    ch = channels_for(brand)
    return ch, (ch[0] if ch else None)


@app.callback(Output("cfg_table", "data"),
              Input("brand", "value"), Input("cfg_regen", "n_clicks"))
def _load_cfg(brand, _regen):
    try:
        from dash import ctx
        regen = ctx.triggered_id == "cfg_regen"
    except Exception:          # outside a live callback (tests)
        regen = False
    cfg = load_or_create_config(brand, regenerate=regen)
    return cfg[CFG_COLS].to_dict("records")


@app.callback(Output("cfg_status", "children"),
              Input("cfg_save", "n_clicks"),
              State("cfg_table", "data"), State("brand", "value"),
              prevent_initial_call=True)
def _save_cfg(_n, rows, brand):
    cfg = pd.DataFrame(rows, columns=CFG_COLS)
    for c in ("adstock_decay", "coef_lower", "coef_upper"):
        cfg[c] = pd.to_numeric(cfg[c], errors="coerce")
    bad = cfg[(cfg.coef_lower.notna()) & (cfg.coef_upper.notna())
              & (cfg.coef_lower > cfg.coef_upper)]
    if len(bad):
        return f"NOT saved: coef_lower > coef_upper for {list(bad.variable)}"
    cfg.to_csv(cfg_path(brand), index=False)
    n_force = int((cfg.role == "force").sum())
    n_excl = int((cfg.role == "exclude").sum())
    return (f"Saved {cfg_path(brand)} — {len(cfg)} variables, "
            f"{n_force} forced, {n_excl} excluded. Next run uses it.")


def _metric(label, value):
    return html.Div(style=CARD, children=[
        html.Div(label, style={"fontSize": "12px", "color": "#4a5568"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "600"}),
    ])


@app.callback(
    Output("metrics", "children"), Output("fig_fit", "figure"),
    Output("fig_resid", "figure"), Output("fig_residhist", "figure"),
    Output("coef_table", "data"), Output("coef_table", "columns"),
    Output("err", "children"),
    Input("run", "n_clicks"),
    State("brand", "value"), State("channel", "value"),
    State("target", "value"), State("window", "value"),
    prevent_initial_call=False)
def run_model(_, brand, channel, target, window):
    empty = go.Figure()
    if not channel:
        return [], empty, empty, empty, [], [], ""
    try:
        load_or_create_config(brand)          # ensure per-brand config exists
        cfg = cp.ModelConfig(target=target, model_weeks=int(window),
                             variable_config=cfg_path(brand))
        r = cp.run_slice("", brand, channel, config=cfg, df=load_sheet(brand))
    except Exception as e:
        return [], empty, empty, empty, [], [], f"Model failed: {e}"

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
    return metrics, f1, f2, f3, coef_rows, cols, ""


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

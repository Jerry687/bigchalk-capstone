"""
Phase 4 MVP: Plotly Dash dashboard for the Big Chalk regression engine.

Run + diagnostics view: pick Brand x Channel x target, set the modeling
window, run the constrained model, inspect fit quality.

    pip install dash
    cd code && python dashboard.py     # http://127.0.0.1:8050

Next milestones (per team plan): variable control panel (roles / signs /
bounds / decays editing variable_config from the UI), contributions
explorer, batch overview page.
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
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update

import capstone_pipeline as cp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
VARCFG = os.path.join(ROOT, "variable_config.csv")

TARGET_CHOICES = ["Volume Sales", "Dollar Sales"]
WINDOW_CHOICES = [52, 104, 156]


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


app = Dash(__name__, title="Big Chalk MMM Engine")

CARD = {"padding": "12px 18px", "borderRadius": "8px", "background": "#f7fafc",
        "border": "1px solid #e2e8f0", "textAlign": "center", "minWidth": "120px"}

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
                    style={"height": "38px", "padding": "0 24px",
                           "background": "#4c51bf", "color": "white",
                           "border": "none", "borderRadius": "6px",
                           "cursor": "pointer"}),
    ]),
    dcc.Loading(children=[
        html.Div(id="metrics", style={"display": "flex", "gap": "12px",
                                      "margin": "18px 0", "flexWrap": "wrap"}),
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
    ]),
    html.Div(id="err", style={"color": "#c53030", "marginTop": "8px"}),
])


@app.callback(Output("channel", "options"), Output("channel", "value"),
              Input("brand", "value"))
def _channels(brand):
    ch = channels_for(brand)
    return ch, (ch[0] if ch else None)


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
        cfg = cp.ModelConfig(
            target=target, model_weeks=int(window),
            force_include=["ACV Weighted Distribution"],
            variable_config=VARCFG if os.path.exists(VARCFG) and target == "Volume Sales" else None,
        )
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

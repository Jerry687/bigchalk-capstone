"""
Big Chalk Mix Engine — Bench UI (Claude Design handoff, option 1a).

Visual system: IBM Plex Sans/Mono, chalk-navy #1E4FA3, left workspace rail,
top slice-picker bar, metric strip with REVIEW flags, family color chips,
health dots on the batch screen. Styling lives in assets/bench.css.

Screens (left rail):
  Diagnostics    — metric strip, actual-vs-fitted with holdout band,
                   residuals, coefficient table with family chips + t-bars.
  Variables      — per-product editable config (sign / adstock / bounds /
                   role); force rows tinted; guardrail banner.
  Contributions  — due-tos by model year, signed avg weekly due-tos,
                   YoY table with % pills.
  Batch          — all models with health dots, review flags, click a row
                   to load that slice; Run all combinations.

    pip install dash
    cd code && python dashboard.py     # http://127.0.0.1:8050
"""
import os
import time
from functools import lru_cache

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx

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
FLAG_THRESHOLD = 40.0    # holdout MAPE % above which a model is flagged

# design tokens (assets/bench.css holds the full set)
INK, INK2, INK3, MUTED = "#101828", "#475467", "#667085", "#98A2B3"
NAVY, WARN, RED, GREEN = "#1E4FA3", "#B54708", "#B42318", "#067647"
GRID = "#EEF1F5"
FAM_CHIP = {
    "Distribution": ("#E0F2EC", "#067647"), "Trade": ("#EBF1FB", "#1E4FA3"),
    "Seasonality": ("#FDF2E3", "#B54708"), "Competitive": ("#FBE9E7", "#B42318"),
    "Macro": ("#EEEAF7", "#5925DC"), "Media": ("#E0F0F6", "#0E7090"),
    "Price": ("#FBE9E7", "#B42318"), "Category Price": ("#F2F4F7", "#475467"),
    "Trend": ("#F2F4F7", "#475467"), "(intercept)": ("#F2F4F7", "#475467"),
}
NAV_ICONS = {
    "diag": "M1.5 8.5h3l2-5 3 9 2-6.5h3",
    "vars": "M2 4.5h7 M12 4.5h2 M9.5 2.5v4 M2 11.5h2 M7 11.5h7 M4.5 9.5v4",
    "contrib": "M2.5 3h8 M2.5 8h11 M2.5 13h5",
    "batch": "M2 2h5v5H2z M9 2h5v5H9z M2 9h5v5H2z M9 9h5v5H9z",
}
SCREENS = [("diag", "Diagnostics"), ("vars", "Variables"),
           ("contrib", "Contributions"), ("batch", "Batch")]


def _slug(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


@lru_cache(maxsize=24)
def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


@lru_cache(maxsize=4)
def detect_product_sheets(path: str) -> tuple:
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
    df = load_sheet(path, sheet)
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c])
           and not str(c).startswith("_C_")]
    pref = [c for c in PREFERRED_TARGETS if c in df.columns]
    # key=str: some client sheets carry numeric column headers, which
    # cannot be sorted against strings directly
    rest = sorted((c for c in num if c not in pref), key=str)
    return pref + rest


def cfg_path(path: str, sheet: str) -> str:
    ds = _slug(os.path.splitext(os.path.basename(path))[0])[:24]
    return os.path.join(CFG_DIR, f"varconfig_{ds}_{_slug(sheet)}.csv")


def batch_summary_path(path: str) -> str:
    ds = _slug(os.path.splitext(os.path.basename(path))[0])[:24]
    return os.path.join(ROOT, "outputs", f"batch_summary_{ds}.csv")


def load_or_create_config(path: str, sheet: str,
                          regenerate: bool = False) -> pd.DataFrame:
    p = cfg_path(path, sheet)
    if os.path.exists(p) and not regenerate:
        return pd.read_csv(p)
    cfg = cp.generate_variable_config(load_sheet(path, sheet))
    cfg.loc[cfg["variable"] == "ACV Weighted Distribution", "role"] = "force"
    cfg.to_csv(p, index=False)
    return cfg


def run_batch(path: str, target_pref: str, window: int) -> pd.DataFrame:
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
                continue
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


# ---------------- formatting helpers ----------------

def fmt_compact(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    sign = "−" if v < 0 else "+"
    if a >= 1e6:
        return f"{sign}{a/1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}{a/1e3:.1f}k"
    return f"{sign}{a:.1f}"


def fam_chip(family: str):
    bg, fg = FAM_CHIP.get(family, ("#F2F4F7", "#475467"))
    return html.Span(family, className="chip",
                     style={"background": bg, "color": fg})


def _fig_base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color=INK2),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=1,
                    xanchor="right", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#D0D5DD",
                     tickfont=dict(family="IBM Plex Mono", size=10, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#D0D5DD",
                     tickfont=dict(family="IBM Plex Mono", size=10, color=MUTED))
    return fig


def card(title, sub, body, pad=True):
    # NB: test `is not None` — an empty Dash component is falsy (len == 0),
    # and dropping it silently invalidates any callback that outputs to it
    head = html.Div([html.Div(title, className="card-title")]
                    + ([html.Div(sub, className="card-sub")]
                       if sub is not None else []),
                    style={"display": "flex", "alignItems": "baseline",
                           "gap": "10px", "marginBottom": "4px"})
    return html.Div([head, body],
                    className="card" + (" card-pad" if pad else ""))


def _icon(path_d, color=INK3):
    return html.Span(
        DangerouslySetSvg(path_d, color),
        style={"display": "flex", "width": "16px", "height": "16px"})


def DangerouslySetSvg(path_d, color):
    # dash html can't inline raw svg easily; use an Img data URL
    parts = "".join(f'<path d="{p.strip()}"/>' for p in path_d.split("M") if p.strip())
    paths = "".join(f'<path d="M{p.strip()}"/>' for p in path_d.split("M") if p.strip())
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" '
           f'width="16" height="16" fill="none" stroke="{color}" '
           f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
           f'{paths}</svg>')
    import base64
    b64 = base64.b64encode(svg.encode()).decode()
    return html.Img(src=f"data:image/svg+xml;base64,{b64}",
                    style={"width": "16px", "height": "16px"})


# ---------------- app & layout ----------------

app = Dash(__name__, title="Big Chalk Mix Engine")

TBL_BASE = dict(
    style_cell={"fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "12.5px",
                "textAlign": "left", "padding": "6px 10px",
                "backgroundColor": "#fff", "color": INK},
    style_header={"fontWeight": "600"},
    style_as_list_view=True,
)


def nav_button(key, label, active=False):
    return html.Button(
        [_icon(NAV_ICONS[key], NAVY if active else INK3), label],
        id=f"nav-{key}", n_clicks=0,
        className="active" if active else "")


def pick(label, dd_id, width="140px", wide=False):
    return html.Div([
        html.Div(label, className="plabel"),
        dcc.Dropdown(id=dd_id, clearable=False,
                     style={"width": width}),
    ], className="bench-pick" + (" wide" if wide else ""))


app.layout = html.Div(className="bench-app", children=[
    dcc.Store(id="datapath"),
    dcc.Store(id="batch_sel"),

    # ═══ left rail ═══
    html.Div(className="bench-rail", children=[
        html.Div(className="bench-logo", children=[
            html.Div("BC", className="bench-logo-mark"),
            html.Div([html.Div("BIG CHALK", className="bench-logo-name"),
                      html.Div("Mix Engine", className="bench-logo-sub")]),
        ]),
        html.Div(className="bench-navwrap", children=[
            html.Div("WORKSPACE", className="bench-navhead"),
            html.Div(className="bench-nav", children=[
                nav_button(k, lbl, active=(k == "diag"))
                for k, lbl in SCREENS]),
        ]),
        html.Div(className="bench-dataset", children=[
            html.Div("DATASET", className="head"),
            html.Div(id="dataset_file", className="file", children="—"),
            html.Div(id="load_status", className="meta",
                     children="Enter a path and load."),
            dcc.Input(id="datafile_input", type="text",
                      value=DEFAULT_DATA if os.path.exists(DEFAULT_DATA) else "",
                      placeholder="C:\\path\\to\\data.xlsx"),
            html.Button("Load data", id="load_data", n_clicks=0,
                        className="btn btn-primary btn-small loadbtn"),
        ]),
    ]),

    # ═══ main column ═══
    html.Div(className="bench-main", children=[

        html.Div(className="bench-topbar", children=[
            html.Div(className="bench-picks", children=[
                pick("PRODUCT", "brand", width="150px"),
                pick("CHANNEL", "channel", width="150px"),
                pick("TARGET", "target", width="210px", wide=True),
                pick("WINDOW", "window", width="120px"),
            ]),
            html.Div(style={"flex": 1}),
            html.Div(id="runstat", className="bench-runstat"),
            html.Button("▶  Run model", id="run", n_clicks=0,
                        className="btn btn-primary"),
        ]),

        # ── Diagnostics ──
        html.Div(id="scr-diag", className="bench-screen", children=[dcc.Loading(children=[
            html.Div(id="err", style={"color": RED, "fontSize": "13px",
                                      "fontWeight": 600}),
            html.Div(id="metrics", className="bench-metrics"),
            html.Div(style={"display": "flex", "gap": "14px",
                            "marginTop": "14px"}, children=[
                html.Div(style={"flex": 2.1}, children=[card(
                    "Actual vs fitted — weekly", html.Span(id="fit_sub"),
                    dcc.Graph(id="fig_fit", config={"displayModeBar": False}))]),
                html.Div(style={"flex": 1, "display": "flex",
                                "flexDirection": "column", "gap": "14px"},
                         children=[
                    card("Residuals vs fitted", None,
                         dcc.Graph(id="fig_resid",
                                   config={"displayModeBar": False})),
                    card("Residual distribution", None,
                         dcc.Graph(id="fig_residhist",
                                   config={"displayModeBar": False})),
                ]),
            ]),
            html.Div(className="card", style={"marginTop": "14px"}, children=[
                html.Div(className="card-head", children=[
                    html.Div("Coefficients", className="card-title"),
                    html.Div(id="coef_sub", className="card-sub"),
                ]),
                html.Div(id="coef_grid"),
            ]),
        ])]),

        # ── Variables ──
        html.Div(id="scr-vars", className="bench-screen",
                 style={"display": "none"}, children=[
            html.Div(style={"display": "flex", "gap": "10px",
                            "alignItems": "center"}, children=[
                html.Div(style={"flex": 1}),
                html.Button("Regenerate defaults", id="cfg_regen", n_clicks=0,
                            className="btn btn-ghost"),
                html.Button("Save config", id="cfg_save", n_clicks=0,
                            className="btn btn-primary",
                            style={"height": "36px"}),
            ]),
            html.Div(id="cfg_status", style={"fontSize": "12px",
                                             "color": INK3}),
            html.Div(className="bench-banner", children=[html.Span([
                html.B("Sign is a guardrail, not a wish. ",
                       style={"fontWeight": 600}),
                "Positive-signed variables can never enter with a negative "
                "coefficient. Role ", html.B("force", style={"fontWeight": 600}),
                " keeps a variable in every run; ",
                html.B("exclude", style={"fontWeight": 600}),
                " bans it. Custom bounds override the sign. "
                "Adstock applies to media only."])]),
            html.Div(className="card", children=[dash_table.DataTable(
                id="cfg_table", editable=True, page_size=200,
                columns=[
                    {"id": "variable", "name": "variable", "editable": False},
                    {"id": "family", "name": "family"},
                    {"id": "sign", "name": "expected sign",
                     "presentation": "dropdown"},
                    {"id": "adstock_decay", "name": "adstock", "type": "numeric"},
                    {"id": "coef_lower", "name": "bound lo", "type": "numeric"},
                    {"id": "coef_upper", "name": "bound hi", "type": "numeric"},
                    {"id": "role", "name": "role", "presentation": "dropdown"},
                ],
                dropdown={
                    "sign": {"options": [{"label": s, "value": s} for s in
                             ["positive", "negative", "unconstrained"]]},
                    "role": {"options": [{"label": s, "value": s} for s in
                             ["auto", "force", "exclude"]]},
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{role} = "force"'},
                     "backgroundColor": "#F5F8FD"},
                    {"if": {"filter_query": '{role} = "exclude"'},
                     "backgroundColor": "#FFF6F5"},
                ],
                **TBL_BASE,
            )]),
        ]),

        # ── Contributions ──
        html.Div(id="scr-contrib", className="bench-screen",
                 style={"display": "none"}, children=[dcc.Loading(children=[
            html.Div(style={"display": "flex", "gap": "14px"}, children=[
                html.Div(style={"flex": 1.2}, children=[card(
                    "Due-to totals by model year", None,
                    dcc.Graph(id="fig_cby",
                              config={"displayModeBar": False}))]),
                html.Div(style={"flex": 1}, children=[card(
                    "Avg weekly due-to per driver",
                    "Signed; media averaged over execution weeks only",
                    dcc.Graph(id="fig_avgc",
                              config={"displayModeBar": False}))]),
            ]),
            html.Div(className="card", style={"marginTop": "14px"}, children=[
                html.Div(className="card-head", children=[
                    html.Div("Due-tos by model year", className="card-title"),
                    html.Div("Decomposes the fitted line exactly: intercept "
                             "+ Σ coefficient × value, per week",
                             className="card-sub"),
                ]),
                html.Div(id="yoy_grid"),
            ]),
        ])]),

        # ── Batch ──
        html.Div(id="scr-batch", className="bench-screen",
                 style={"display": "none"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "10px"}, children=[
                html.Div("All models", style={"fontSize": "15px",
                                              "fontWeight": 600}),
                html.Div(id="batch_pills", style={"display": "flex",
                                                  "gap": "8px"}),
                html.Div(style={"flex": 1}),
                html.Button("Reload summary", id="batch_refresh", n_clicks=0,
                            className="btn btn-ghost"),
                html.Button("Run all combinations", id="batch_run", n_clicks=0,
                            className="btn btn-primary",
                            style={"height": "36px"}),
            ]),
            html.Div("Every product × channel of the loaded datafile, using "
                     "each product's saved config. Click a row to load that "
                     "slice into the controls, then hit Run model.",
                     style={"fontSize": "12px", "color": INK3}),
            dcc.Loading(children=[html.Div(className="card", children=[
                dash_table.DataTable(
                    id="batch_table", sort_action="native",
                    filter_action="native", row_selectable="single",
                    page_size=100, **TBL_BASE,
                )])]),
        ]),
    ]),
])


# ---------------- callbacks ----------------

@app.callback(
    [Output(f"scr-{k}", "style") for k, _ in SCREENS]
    + [Output(f"nav-{k}", "className") for k, _ in SCREENS],
    [Input(f"nav-{k}", "n_clicks") for k, _ in SCREENS])
def _nav(*_):
    try:
        trig = ctx.triggered_id
    except Exception:          # outside a live callback (tests)
        trig = None
    active = (trig or "nav-diag").replace("nav-", "")
    styles = [{} if k == active else {"display": "none"} for k, _ in SCREENS]
    classes = ["active" if k == active else "" for k, _ in SCREENS]
    return styles + classes


@app.callback(Output("datapath", "data"), Output("load_status", "children"),
              Output("dataset_file", "children"),
              Input("load_data", "n_clicks"),
              State("datafile_input", "value"))
def _load_data(_n, path):
    path = (path or "").strip().strip('"')
    if not path:
        return None, "Enter a path to an .xlsx datafile.", "—"
    if not os.path.exists(path):
        return None, f"File not found: {path}", "—"
    try:
        sheets = detect_product_sheets(path)
    except Exception as e:
        return None, f"Could not read workbook: {e}", "—"
    if not sheets:
        return None, ("No product sheets found (need Geography + Time "
                      "columns, or sheets named 'Brand *')."), "—"
    n_ch = len(channels_for(path, sheets[0]))
    n_wk = len(load_sheet(path, sheets[0])) // max(n_ch, 1)
    meta = f"{len(sheets)} products · ~{n_ch} channels · ~{n_wk} weeks each"
    return ({"path": path, "sheets": list(sheets)}, meta,
            os.path.basename(path))


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


@app.callback(Output("window", "options"), Output("window", "value"),
              Input("datapath", "data"))
def _window(dp):
    return [{"label": f"{w} wks", "value": w} for w in WINDOW_CHOICES], 104


@app.callback(Output("cfg_table", "data"),
              Input("brand", "value"), Input("cfg_regen", "n_clicks"),
              State("datapath", "data"))
def _load_cfg(brand, _regen, dp):
    if not dp or not brand:
        return []
    try:
        regen = ctx.triggered_id == "cfg_regen"
    except Exception:
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
        return f"NOT saved: bound lo > bound hi for {list(bad.variable)}"
    cfg.to_csv(cfg_path(dp["path"], brand), index=False)
    return (f"Saved — {len(cfg)} variables, "
            f"{int((cfg.role == 'force').sum())} forced, "
            f"{int((cfg.role == 'exclude').sum())} excluded. "
            "Next run uses it.")


def _summary_for(dp):
    if dp and os.path.exists(batch_summary_path(dp["path"])):
        return batch_summary_path(dp["path"])
    if dp and os.path.normpath(dp["path"]) == os.path.normpath(DEFAULT_DATA) \
            and os.path.exists(SUMMARY):
        return SUMMARY
    return None


def _batch_payload(s):
    s = s.copy()
    s.insert(0, "health", "●")
    cols = ([{"name": "", "id": "health"}]
            + [{"name": c.replace("_", " "), "id": c}
               for c in s.columns if c != "health"])
    flagged = int((s["MAPE_holdout_pct"] > FLAG_THRESHOLD).sum())
    pills = [
        html.Span(f"{len(s)} models", className="pill"),
        html.Span(["median R² ", html.Span(f"{s.R2.median():.2f}",
                                           className="mono")],
                  className="pill"),
        html.Span(["median holdout MAPE ",
                   html.Span(f"{s.MAPE_holdout_pct.median():.1f}%",
                             className="mono")], className="pill"),
    ]
    if flagged:
        pills.append(html.Span(f"{flagged} need review",
                               className="pill pill-red"))
    styles = [
        {"if": {"filter_query": f"{{MAPE_holdout_pct}} > {FLAG_THRESHOLD}"},
         "backgroundColor": "#FFF6F5"},
        {"if": {"column_id": "health",
                "filter_query": f"{{MAPE_holdout_pct}} > {FLAG_THRESHOLD}"},
         "color": "#D92D20"},
        {"if": {"column_id": "health",
                "filter_query": f"{{MAPE_holdout_pct}} <= {FLAG_THRESHOLD} && "
                                "{MAPE_holdout_pct} > 10"},
         "color": "#F79009"},
        {"if": {"column_id": "health",
                "filter_query": "{MAPE_holdout_pct} <= 10"},
         "color": "#12B76A"},
    ]
    return s.round(3).to_dict("records"), cols, pills, styles


@app.callback(Output("batch_table", "data"), Output("batch_table", "columns"),
              Output("batch_pills", "children"),
              Output("batch_table", "style_data_conditional"),
              Input("batch_refresh", "n_clicks"), Input("datapath", "data"))
def _load_batch(_n, dp):
    p = _summary_for(dp)
    if p is None:
        return [], [], [html.Span("No batch yet — hit Run all combinations.",
                                  className="pill")], []
    return _batch_payload(pd.read_csv(p))


@app.callback(Output("batch_table", "data", allow_duplicate=True),
              Output("batch_table", "columns", allow_duplicate=True),
              Output("batch_pills", "children", allow_duplicate=True),
              Output("batch_table", "style_data_conditional",
                     allow_duplicate=True),
              Input("batch_run", "n_clicks"),
              State("datapath", "data"), State("target", "value"),
              State("window", "value"), prevent_initial_call=True)
def _run_batch(_n, dp, target, window):
    if not dp:
        return [], [], [html.Span("Load a datafile first.",
                                  className="pill")], []
    t0 = time.time()
    s = run_batch(dp["path"], target or "Volume Sales", window or 104)
    data, cols, pills, styles = _batch_payload(s)
    pills.append(html.Span(f"ran in {time.time()-t0:.0f}s", className="pill"))
    return data, cols, pills, styles


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


def metric_card(label, value, unit=None, warn=False, chip=None):
    lab = [label] + ([html.Span(chip, className="review-chip")] if chip else [])
    val = [value] + ([html.Span(unit, className="munit")] if unit else [])
    return html.Div([html.Div(lab, className="mlabel"),
                     html.Div(val, className="mvalue")],
                    className="mcard" + (" warn" if warn else ""))


COEF_GRID = "2.6fr 1fr 0.9fr 1.2fr 1.1fr 1.2fr"
YOY_GRID = "2.4fr 1fr 1fr 1fr 0.8fr"


def coef_grid_rows(r, fit):
    head = html.Div([html.Div(x) for x in
                     ["VARIABLE", "FAMILY", "SIGN", "T-STAT",
                      html.Div("COEFFICIENT", style={"textAlign": "right"}),
                      html.Div("AVG WKLY DUE-TO", style={"textAlign": "right"})]],
                    className="bgrid-head",
                    style={"gridTemplateColumns": COEF_GRID})
    sign_sym = {"positive": ("+", GREEN), "negative": ("−", RED),
                "unconstrained": ("~", INK2)}
    rows = [head]
    for name in ["const"] + r["selected"]:
        spec = r["specs_by_name"].get(name)
        family = spec.family if spec else "(intercept)"
        sym, scolor = sign_sym.get(spec.sign, ("—", INK2)) if spec else ("—", INK2)
        t = fit.tstats.get(name, np.nan) if name != "const" else np.nan
        t_abs = 0 if np.isnan(t) else abs(t)
        due = (r["avg_contrib"].loc["Intercept" if name == "const" else name,
                                    "avg_weekly_contribution"])
        namecell = [html.Span("const (intercept)" if name == "const" else name,
                              style={"overflow": "hidden",
                                     "textOverflow": "ellipsis",
                                     "whiteSpace": "nowrap"})]
        if name in r["forced"]:
            namecell.append(html.Span("FORCED", className="forced-chip"))
        rows.append(html.Div([
            html.Div(namecell, style={"display": "flex",
                                      "alignItems": "center",
                                      "overflow": "hidden",
                                      "fontWeight": 500}),
            html.Div(fam_chip(family)),
            html.Div(sym, className="mono", style={"color": scolor}),
            html.Div([
                html.Span(html.Span(style={
                    "width": f"{min(100, t_abs/10*100):.0f}%",
                    "background": NAVY if t_abs >= 2 else "#D0D5DD"}),
                    className="tbar"),
                html.Span("—" if np.isnan(t) else f"{t:.2f}",
                          className="mono",
                          style={"fontSize": "11.5px", "color": INK2,
                                 "marginLeft": "7px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(f"{fit.coef[name]:,.2f}", className="mono",
                     style={"textAlign": "right", "fontSize": "12px"}),
            html.Div(fmt_compact(due), className="mono",
                     style={"textAlign": "right", "fontSize": "12px",
                            "color": RED if due < 0 else GREEN}),
        ], className="bgrid", style={"gridTemplateColumns": COEF_GRID}))
    return rows


def yoy_grid_rows(cby):
    year_cols = [c for c in cby.columns if not str(c).startswith("YoY")]
    head_cells = [html.Div("DRIVER")] + \
        [html.Div(str(c).upper(), style={"textAlign": "right"})
         for c in year_cols] + \
        [html.Div("YOY Δ", style={"textAlign": "right"}),
         html.Div("YOY %", style={"textAlign": "right"})]
    n_years = len(year_cols)
    grid = f"2.4fr {'1fr ' * n_years}1fr 0.8fr"
    rows = [html.Div(head_cells, className="bgrid-head",
                     style={"gridTemplateColumns": grid})]
    for driver, rr in cby.iterrows():
        chg = rr.get("YoY_change", np.nan)
        pct = rr.get("YoY_pct", np.nan)
        cells = [html.Div(str(driver), style={"fontWeight": 500})]
        for c in year_cols:
            cells.append(html.Div(f"{rr[c]:,.0f}", className="mono",
                                  style={"textAlign": "right",
                                         "fontSize": "12px"}))
        chg_color = MUTED if (np.isnan(chg) or chg == 0) \
            else (RED if chg < 0 else GREEN)
        cells.append(html.Div(
            "—" if np.isnan(chg) else f"{chg:+,.0f}".replace("-", "−"),
            className="mono", style={"textAlign": "right", "fontSize": "12px",
                                     "color": chg_color}))
        if np.isnan(pct):
            pill = html.Span("—", className="pct-pill",
                             style={"background": "#F2F4F7", "color": INK3})
        else:
            bg, fg = (("#E0F2EC", GREEN) if pct > 0 else
                      ("#FBE9E7", RED) if pct < 0 else ("#F2F4F7", INK3))
            pill = html.Span(f"{pct:+.1f}%", className="pct-pill",
                             style={"background": bg, "color": fg})
        cells.append(html.Div(pill, style={"textAlign": "right"}))
        rows.append(html.Div(cells, className="bgrid",
                             style={"gridTemplateColumns": grid}))
    return rows


@app.callback(
    Output("metrics", "children"), Output("fig_fit", "figure"),
    Output("fit_sub", "children"),
    Output("fig_resid", "figure"), Output("fig_residhist", "figure"),
    Output("coef_grid", "children"), Output("coef_sub", "children"),
    Output("fig_cby", "figure"), Output("fig_avgc", "figure"),
    Output("yoy_grid", "children"),
    Output("err", "children"), Output("runstat", "children"),
    Input("run", "n_clicks"),
    State("brand", "value"), State("channel", "value"),
    State("target", "value"), State("window", "value"),
    State("datapath", "data"),
    prevent_initial_call=False)
def run_model(_, brand, channel, target, window, dp):
    empty = _fig_base(go.Figure(), 260)
    blank = ([], empty, "", empty, empty, [], "", empty, empty, [])
    missing = [lbl for lbl, v in
               [("datafile (hit Load data in the rail)", dp),
                ("product", brand), ("channel", channel),
                ("target", target), ("window", window)] if not v]
    if missing:
        msg = "Cannot run — missing: " + ", ".join(missing)
        print(f"[run_model] {msg}", flush=True)
        return (*blank, msg, "")
    path = dp["path"]
    t0 = time.time()
    print(f"[run_model] {brand} x {channel} · target={target} "
          f"window={window}", flush=True)
    try:
        load_or_create_config(path, brand)
        cfg = cp.ModelConfig(target=target, model_weeks=int(window),
                             variable_config=cfg_path(path, brand))
        r = cp.run_slice("", brand, channel, config=cfg,
                         df=load_sheet(path, brand))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (*blank, f"Model failed: {type(e).__name__}: {e}",
                "run failed — see console")
    dur = time.time() - t0

    fit, df, split = r["fit"], r["df"], r["split"]
    y = r["y"]
    ho = r["holdout_mape"]
    ho_warn = (not np.isnan(ho)) and ho > FLAG_THRESHOLD

    metrics = [
        metric_card("R²", f"{fit.r2:.3f}"),
        metric_card("ADJ R²", f"{fit.adj_r2:.3f}"),
        metric_card("IN-SAMPLE MAPE", f"{fit.mape:.1f}", unit="%"),
        metric_card("HOLDOUT MAPE",
                    "n/a" if np.isnan(ho) else f"{ho:.1f}",
                    unit=None if np.isnan(ho) else "%",
                    warn=ho_warn, chip="REVIEW" if ho_warn else None),
        metric_card("DURBIN–WATSON", f"{fit.meta['durbin_watson']:.2f}"),
        metric_card("PREDICTORS", f"{len(r['selected'])}",
                    unit=f" · {len(r['forced'])} forced"
                    if r["forced"] else None),
    ]

    f1 = go.Figure()
    if len(r["pred_te"]):
        f1.add_vrect(x0=df["date"].iloc[split], x1=df["date"].iloc[-1],
                     fillcolor="#FEF3E2", opacity=0.6, line_width=0)
    f1.add_scatter(x=df["date"], y=y, name="Actual",
                   line=dict(color=INK, width=1.8))
    f1.add_scatter(x=df["date"].iloc[:split], y=fit.fitted, name="Fitted",
                   line=dict(color=NAVY, width=1.7, dash="dash"))
    if len(r["pred_te"]):
        f1.add_scatter(x=df["date"].iloc[split:], y=r["pred_te"],
                       name="Holdout forecast",
                       line=dict(color=WARN, width=1.7, dash="dash"))
        f1.add_vline(x=df["date"].iloc[split], line_dash="dot",
                     line_color="#D0D5DD")
    _fig_base(f1, 300)

    f2 = go.Figure(go.Scatter(x=fit.fitted, y=fit.resid, mode="markers",
                              marker=dict(color=NAVY, opacity=0.42, size=6)))
    f2.add_hline(y=0, line_dash="dash", line_color="#D0D5DD")
    _fig_base(f2, 140)
    f3 = go.Figure(go.Histogram(x=fit.resid, marker_color="#B7CBEA"))
    _fig_base(f3, 140)

    coef_children = coef_grid_rows(r, fit)
    n_cand = len(r["specs"])
    conflicts = len(r["sign_conflicts"])
    coef_sub = (f"{len(r['selected'])} selected of {n_cand} candidates · "
                + ("all pass sign checks" if conflicts == 0
                   else f"{conflicts} sign conflicts"))

    cby = r["contrib_by_year"]
    year_cols = [c for c in cby.columns if not str(c).startswith("YoY")]
    plot_tbl = cby[year_cols].drop("Intercept", errors="ignore")
    f4 = go.Figure()
    palette = ["#C5CEDC", NAVY, "#0E7090"]
    for i, yc in enumerate(year_cols):
        f4.add_bar(y=plot_tbl.index, x=plot_tbl[yc], name=str(yc),
                   orientation="h", marker_color=palette[i % len(palette)])
    f4.update_layout(barmode="group")
    _fig_base(f4, max(320, 30 * len(plot_tbl)))

    avg = (r["avg_contrib"]["avg_weekly_contribution"]
           .drop("Intercept", errors="ignore").sort_values())
    f5 = go.Figure(go.Bar(
        y=avg.index, x=avg.values, orientation="h",
        marker_color=[("#D9705E" if v < 0 else NAVY) for v in avg.values]))
    _fig_base(f5, max(320, 30 * len(avg)))

    yoy_children = yoy_grid_rows(cby.round(0))
    fit_sub = f"{brand} × {channel} · {target}"
    runstat = [f"Config saved · last run ",
               html.Span(time.strftime("%H:%M"), className="mono"),
               f" ({dur:.1f}s)"]
    return (metrics, f1, fit_sub, f2, f3, coef_children, coef_sub,
            f4, f5, yoy_children, "", runstat)


def _validate_wiring():
    """Fail loudly at startup if any callback references a component id
    missing from the layout — the browser otherwise disables the callback
    silently when debug=False (bitten once by a falsy empty component)."""
    import collections
    import re as _re
    ids = []

    def walk(c):
        if getattr(c, "id", None) is not None:
            ids.append(c.id)
        ch = getattr(c, "children", None)
        if ch is None:
            return
        for x in (ch if isinstance(ch, (list, tuple)) else [ch]):
            if hasattr(x, "to_plotly_json"):
                walk(x)

    walk(app.layout)
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    missing = set()
    for spec in app._callback_list:
        for dep in spec["inputs"] + spec.get("state", []):
            if dep["id"] not in ids:
                missing.add(dep["id"])
        for oid in _re.findall(r"([\w-]+)\.[\w]+", spec["output"]):
            if oid not in ids:
                missing.add(oid)
    if dupes or missing:
        raise RuntimeError(
            f"Callback wiring broken - duplicate ids: {dupes or 'none'}; "
            f"ids referenced but missing from layout: {sorted(missing) or 'none'}")
    print(f"wiring OK: {len(ids)} components, "
          f"{len(app._callback_list)} callbacks", flush=True)


if __name__ == "__main__":
    _validate_wiring()
    app.run(debug=False, host="127.0.0.1", port=8050)

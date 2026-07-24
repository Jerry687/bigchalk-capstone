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
import shutil
import time
import uuid
from functools import lru_cache

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (Dash, dcc, html, dash_table, Input, Output, State, ctx,
                  no_update)

import capstone_pipeline as cp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(ROOT, "Anonymized Data for Project.xlsx")
CFG_DIR = os.path.join(ROOT, "configs")
SUMMARY = os.path.join(ROOT, "outputs", "all_models_summary.csv")
os.makedirs(CFG_DIR, exist_ok=True)

WINDOW_CHOICES = [52, 104, 156]
CFG_COLS = ["variable", "family", "sign", "adstock_decay",
            "coef_lower", "coef_upper", "role"]
# Canonical variable-template columns (V2 download/upload; also the export's
# "model fit structure" sheet). channel = a specific channel, or ALL = the
# Product default. Priority on run: Product × Channel override > Product default.
TEMPLATE_COLS = ["product", "channel", "variable", "family", "sign", "role",
                 "coef_lower", "coef_upper", "adstock_decay"]
PREFERRED_TARGETS = ["Volume Sales", "Dollar Sales", "Unit Sales"]
FLAG_THRESHOLD = 40.0    # holdout MAPE % above which a model is flagged
VIF_WARN = 10.0          # VIF above which collinearity is flagged (non-blocking)
# Every config-file mutation (save / upload / regenerate / reset / default
# creation) goes through this one lock — shared with capstone_pipeline so app
# writes and default creation can't interleave. Process-local (see cp).
_CFG_WRITE_LOCK = cp._CFG_WRITE_LOCK

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
    "export": "M8 2v8 M4.5 6.5l3.5 3.5 3.5-3.5 M2.5 13.5h11",
    "defs": "M3 2.5h7a2 2 0 012 2v9 M13 2.5H6a2 2 0 00-2 2v9 M3 13.5h10",
}
# key "batch" is kept for the component ids; only the display label changed
# to "Model Runs" per Alex (slide 4 nav rename).
SCREENS = [("diag", "Diagnostics"), ("vars", "Variables"),
           ("contrib", "Contributions"), ("batch", "Model Runs"),
           ("export", "Export"), ("defs", "Definitions")]


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


# Config-path helpers delegate to capstone_pipeline so the dashboard, the
# batch runner, and the CLI runners all resolve the SAME files (single source
# of truth). resolve_cfg_path falls back to the default path (which the
# dashboard's load_or_create_config always creates) so the editor never gets
# a None path.
def cfg_path(path: str, sheet: str) -> str:
    """Product DEFAULT config (channel = ALL)."""
    return cp.default_config_path(path, sheet)


def override_cfg_path(path: str, sheet: str, channel: str) -> str:
    """Product × Channel OVERRIDE config."""
    return cp.override_config_path(path, sheet, channel)


def resolve_cfg_path(path: str, sheet: str, channel: str) -> str:
    """Runtime resolution: Product × Channel override wins if it exists, else
    the Product default. (Jerry's two-tier model.)"""
    return cp.resolve_config_path(path, sheet, channel) or cfg_path(path, sheet)


def save_cfg_path(path: str, sheet: str, scope: str, channel: str) -> str:
    """Where edits are written: 'channel' scope → this channel's override
    file; otherwise → the Product default."""
    if scope == "channel" and channel:
        return override_cfg_path(path, sheet, channel)
    return cfg_path(path, sheet)


def load_cfg_for_edit(path: str, sheet: str, scope: str, channel: str):
    """Config to show in the editor for the chosen scope.
    Returns (df, override_state) where override_state is:
      None       — editing the Product default
      'existing' — editing an existing channel override
      'inherit'  — channel scope with no override yet (seeded from default)."""
    if scope == "channel" and channel:
        ov = override_cfg_path(path, sheet, channel)
        if os.path.exists(ov):
            return pd.read_csv(ov), "existing"
        return load_or_create_config(path, sheet), "inherit"
    return load_or_create_config(path, sheet), None


# ---------------- variable template (V2 download / upload) ----------------

def _cfg_to_template_rows(cfg_df, product, channel):
    return [{"product": product, "channel": channel, "variable": r["variable"],
             "family": r["family"], "sign": r["sign"], "role": r["role"],
             "coef_lower": r["coef_lower"], "coef_upper": r["coef_upper"],
             "adstock_decay": r["adstock_decay"]}
            for _, r in cfg_df.iterrows()]


def build_template_df(path, brand) -> pd.DataFrame:
    """Editable template for a product: the Product default (channel = ALL)
    plus every existing Channel override, in the canonical upload format."""
    rows = _cfg_to_template_rows(load_or_create_config(path, brand), brand, "ALL")
    for channel in channels_for(path, brand):
        ov = override_cfg_path(path, brand, channel)
        if os.path.exists(ov):
            rows += _cfg_to_template_rows(pd.read_csv(ov), brand, channel)
    return pd.DataFrame(rows, columns=TEMPLATE_COLS)


def _norm_channel(channel) -> str:
    if channel is None or (isinstance(channel, float) and np.isnan(channel)):
        return "ALL"
    c = str(channel).strip()
    return "ALL" if c == "" or c.upper() == "ALL" else c


def _tmpl_pick(t, col, default):
    return t[col] if col in t and pd.notna(t[col]) else default


def apply_template_df(tdf, path):
    """Apply an uploaded template ATOMICALLY: validate every (product, channel)
    group first and write NOTHING unless all pass, so a later failure can't
    leave a partial edit. Each group → a config file (channel ALL → Product
    default, else the Channel override); variables in the data NOT listed for a
    group become role=exclude (Alex). Returns (status, [(product, channel,
    detail), …]) where status is 'applied' / 'rejected' / 'rollback_incomplete'
    ('rollback_incomplete' means disk changed but couldn't be fully restored —
    the caller must refresh AND surface the recovery path)."""
    tdf = tdf.copy()
    tdf.columns = [str(c).strip().lower() for c in tdf.columns]
    if not {"product", "channel", "variable"} <= set(tdf.columns):
        return "rejected", [("—", "—", "missing required columns (need product, "
                             "channel, variable)")]

    sheets = set(detect_product_sheets(path))
    # Normalize product / variable / channel BEFORE grouping so that "ALL"/
    # "all", channel case, and stray whitespace collapse into one group instead
    # of forming duplicate groups that overwrite the same file. Channel is
    # canonicalized to the product's actual channel name (its override path is
    # case-folded, so two spellings would otherwise collide).
    _chan_map = {}

    def canon_channel(product, ch):
        ch = _norm_channel(ch)
        if ch == "ALL" or product not in sheets:
            return ch
        if product not in _chan_map:
            _chan_map[product] = {str(c).strip().lower(): c
                                  for c in channels_for(path, product)}
        return _chan_map[product].get(ch.lower(), ch)

    tdf["product"] = tdf["product"].map(lambda x: str(x).strip())
    tdf["variable"] = tdf["variable"].map(lambda x: str(x).strip())
    tdf["channel"] = [canon_channel(p, c)
                      for p, c in zip(tdf["product"], tdf["channel"])]

    plans, errors = [], []          # plans: (product, ch, target_path, cfg_df)
    for (product, ch), grp in tdf.groupby(["product", "channel"], dropna=False):
        product, ch = str(product), str(ch)
        tag = (product, ch)
        if product not in sheets:
            errors.append((*tag, f"unknown product '{product}'"))
            continue
        if ch != "ALL" and ch not in set(channels_for(path, product)):
            errors.append((*tag, f"unknown channel '{ch}' for {product}"))
            continue
        base = cp.generate_default_config(load_sheet(path, product))
        universe = {str(v) for v in base["variable"]}
        listed_names = list(grp["variable"])
        dups = sorted({v for v in listed_names if listed_names.count(v) > 1})
        if dups:
            errors.append((*tag, f"duplicate variables: {dups}"))
            continue
        unknown = [v for v in listed_names if v not in universe]
        if unknown:      # typos -> reject (never silently exclude everything)
            errors.append((*tag, f"unknown variable(s): {unknown[:5]}"
                           + (" …" if len(unknown) > 5 else "")))
            continue
        listed = {str(r["variable"]): r for _, r in grp.iterrows()}
        rows = []
        for _, b in base.iterrows():
            v = str(b["variable"])
            if v in listed:
                t = listed[v]
                rows.append({"variable": v,
                             "family": _tmpl_pick(t, "family", b["family"]),
                             "sign": _tmpl_pick(t, "sign", b["sign"]),
                             "adstock_decay": _tmpl_pick(t, "adstock_decay",
                                                         b["adstock_decay"]),
                             "coef_lower": _tmpl_pick(t, "coef_lower",
                                                      b["coef_lower"]),
                             "coef_upper": _tmpl_pick(t, "coef_upper",
                                                      b["coef_upper"]),
                             "role": _tmpl_pick(t, "role", b["role"])})
            else:                                   # unlisted -> excluded
                row = {k: b[k] for k in CFG_COLS}
                row["role"] = "exclude"
                rows.append(row)
        ok, cfg, bad = _validate_cfg_rows(rows)       # validate, don't write
        if not ok:
            errors.append((*tag, f"bound lo >= hi for {bad}"))
            continue
        target = (cfg_path(path, product) if ch == "ALL"
                  else override_cfg_path(path, product, ch))
        plans.append((product, ch, target, cfg))

    if errors:                       # atomic: reject the whole upload
        skipped = [(p, c, "not written (upload rejected)") for p, c, _, _ in plans]
        return "rejected", errors + skipped
    if not plans:                    # empty / no-op upload writes nothing
        return "rejected", [("—", "—", "no valid rows to write (empty template)")]

    return _commit_configs_atomic(plans)


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _commit_configs_atomic(plans):
    """Write configs as a best-effort multi-file transaction, serialized by
    _CFG_WRITE_LOCK so concurrent uploads/saves can't interleave on a target.
      1. Stage every config to a UNIQUE temp file. Any staging failure (I/O OR
         serialization) → delete all temps (incl. the partial one, tracked
         before the write) and commit nothing.
      2. Commit: back up each existing target, then os.replace(tmp → target).
         If any commit step fails, ROLL BACK — restore backed-up originals,
         remove newly-created targets. A restore that ITSELF fails is reported
         and its backup is PRESERVED (never delete the last copy of an original)
         with an explicit 'rollback incomplete' status.
    os.replace is atomic per file; cross-file atomicity isn't OS-guaranteed, so
    this backup/rollback keeps the set consistent (single process)."""
    with _CFG_WRITE_LOCK:
        staged = []   # dicts: tmp, target, existed, cfg, product, ch, committed
        try:
            for product, ch, target, cfg in plans:
                tmp = f"{target}.{uuid.uuid4().hex}.uploadtmp"
                staged.append({"tmp": tmp, "target": target,
                               "existed": os.path.exists(target), "cfg": cfg,
                               "product": product, "ch": ch, "committed": False})
                cfg.to_csv(tmp, index=False)       # tmp tracked before write
        except Exception as e:                     # I/O or serialization
            _rm([s["tmp"] for s in staged])
            return "rejected", [("—", "—",
                                 f"write failed — nothing committed: {e}")]

        backups = {}          # target -> backup path
        made_backups = []     # every bak path attempted (for cleanup)
        try:
            for s in staged:
                if s["existed"]:
                    bak = f"{s['target']}.{uuid.uuid4().hex}.bak"
                    made_backups.append(bak)       # tracked before copy
                    shutil.copy2(s["target"], bak)
                    backups[s["target"]] = bak
                os.replace(s["tmp"], s["target"])
                s["committed"] = True
        except Exception as e:
            rollback_errors, keep, failed = [], set(), set()
            for s in staged:
                if not s["committed"]:
                    continue
                tgt, bak = s["target"], backups.get(s["target"])
                if s["existed"] and bak:           # restore original
                    try:
                        os.replace(bak, tgt)       # success -> back to original
                    except Exception as re:
                        rollback_errors.append(f"{tgt}: restore failed ({re})")
                        keep.add(bak)              # KEEP — only surviving original
                        failed.add(tgt)            # this one really stayed changed
                elif not s["existed"]:             # remove newly-created target
                    try:
                        os.remove(tgt)
                    except Exception as re:
                        rollback_errors.append(f"{tgt}: cleanup failed ({re})")
                        failed.add(tgt)            # new file couldn't be removed
            _rm([s["tmp"] for s in staged]
                + [b for b in made_backups if b not in keep])
            if rollback_errors:
                changed = [(s["product"], s["ch"], "CHANGED (not restored)")
                           for s in staged if s["target"] in failed]
                return "rollback_incomplete", changed + [
                    ("—", "—", "commit failed and ROLLBACK INCOMPLETE: "
                     f"{e}. {'; '.join(rollback_errors)}. Original(s) preserved "
                     f"at: {sorted(keep)}")]
            return "rejected", [("—", "—", "commit failed — rolled back to "
                                 f"previous configs: {e}")]

        _rm(made_backups)                          # success — drop backups
        return "applied", [(s["product"], s["ch"],
                            f"written ({len(s['cfg'])} vars, "
                            f"{int((s['cfg'].role == 'exclude').sum())} excluded)")
                           for s in staged]


# ---------------- export workbook (X1) ----------------

def _slice_export_rows(path, brand, channel, target, window):
    """Run one slice with its RESOLVED (override>default) config and return
    (fit-stat row, fit-structure rows [template format], weekly due-to rows)."""
    load_or_create_config(path, brand)      # ensure the default exists (new data)
    cfg_file = resolve_cfg_path(path, brand, channel)
    cfg = cp.ModelConfig(target=target, model_weeks=int(window),
                         variable_config=cfg_file)
    r = cp.run_slice("", brand, channel, config=cfg, df=load_sheet(path, brand))
    fit, ho = r["fit"], r["holdout_mape"]
    stat = {"product": brand, "channel": channel, "target": target,
            "window_weeks": int(window), "R2": round(fit.r2, 4),
            "adj_R2": round(fit.adj_r2, 4), "MAPE_in_pct": round(fit.mape, 2),
            "MAPE_holdout_pct": None if np.isnan(ho) else round(ho, 2),
            "grade": _grade_label(ho), "n_selected": len(r["selected"]),
            "n_forced": len(r["forced"])}
    # fit structure = the config actually used, in template format (so an
    # exported model can be tweaked in Excel and re-uploaded). Tagged with the
    # specific channel; re-uploading it materializes that channel's override.
    struct = _cfg_to_template_rows(pd.read_csv(cfg_file), brand, channel)
    # weekly due-tos — actual per-week (date) contributions, long/tidy format.
    # The reported model spans ALL weeks in the window, so use all dates.
    contrib = fit.contributions.reset_index(drop=True)
    dates = r["df"]["date"].reset_index(drop=True)
    weekly = [{"product": brand, "channel": channel,
               "date": dates.iloc[i] if i < len(dates) else None,
               "driver": drv, "due_to": float(contrib.iloc[i][drv])}
              for i in range(len(contrib)) for drv in contrib.columns]
    return stat, struct, weekly


GRADE_FILL = {"Good Model": GREEN, "Moderate Model": WARN, "Bad Model": RED,
              "No holdout": "#98A2B3"}


def build_export_workbook(path, target, window, mode, brand=None, channel=None):
    """Assemble the export: fit_statistics + fit_structure + weekly_due_tos
    (+ errors) for one slice or every slice. Slice enumeration AND per-slice
    runs are guarded so a missing target / bad product lands in the errors
    sheet instead of crashing. Returns (writer_fn, n_ok, n_err)."""
    stats, structs, weekly, errors = [], [], [], []
    slices = []
    if mode == "slice":
        slices = [(brand, channel)]
    else:
        for sheet in detect_product_sheets(path):
            try:
                dfx = load_sheet(path, sheet)
                if target not in dfx.columns:
                    errors.append({"product": sheet, "channel": "*",
                                   "error": f"target '{target}' not in product"})
                    continue
                for ch in channels_for(path, sheet):
                    if dfx.loc[dfx["Geography"] == ch, target].abs().sum() == 0:
                        continue                # not sold in channel — skip
                    slices.append((sheet, ch))
            except Exception as e:
                errors.append({"product": sheet, "channel": "*",
                               "error": f"{type(e).__name__}: {e}"})

    for br, ch in slices:
        try:
            s, st, wk = _slice_export_rows(path, br, ch, target, window)
            stats.append(s)
            structs.extend(st)
            weekly.extend(wk)
        except Exception as e:
            errors.append({"product": br, "channel": ch,
                           "error": f"{type(e).__name__}: {e}"})

    def _write(buf):
        from openpyxl.styles import Font, PatternFill
        stat_df = (pd.DataFrame(stats) if stats
                   else pd.DataFrame(columns=["product", "channel", "grade"]))
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            stat_df.to_excel(xw, sheet_name="fit_statistics", index=False)
            pd.DataFrame(structs, columns=TEMPLATE_COLS).to_excel(
                xw, sheet_name="fit_structure", index=False)
            (pd.DataFrame(weekly) if weekly else pd.DataFrame(
                columns=["product", "channel", "date", "driver",
                         "due_to"])).to_excel(
                xw, sheet_name="weekly_due_tos", index=False)
            if errors:
                pd.DataFrame(errors).to_excel(xw, sheet_name="errors",
                                              index=False)
            # green/amber/red fill on the grade column (Alex: slide 6)
            if stats and "grade" in stat_df.columns:
                ws = xw.sheets["fit_statistics"]
                gcol = list(stat_df.columns).index("grade") + 1
                for i, val in enumerate(stat_df["grade"], start=2):
                    hexc = GRADE_FILL.get(val)
                    if not hexc:
                        continue
                    cell = ws.cell(row=i, column=gcol)
                    cell.fill = PatternFill("solid", fgColor=hexc.lstrip("#"))
                    cell.font = Font(color="FFFFFF", bold=True)
    return _write, len(stats), len(errors)


def batch_summary_path(path: str) -> str:
    ds = _slug(os.path.splitext(os.path.basename(path))[0])[:24]
    return os.path.join(ROOT, "outputs", f"batch_summary_{ds}.csv")


def fresh_default_df(path: str, sheet: str) -> pd.DataFrame:
    """Freshly auto-generate a config from the data (no file write). Delegates
    to the shared generator so ACV force-include lives in one place."""
    return cp.generate_default_config(load_sheet(path, sheet))


def load_or_create_config(path: str, sheet: str,
                          regenerate: bool = False) -> pd.DataFrame:
    # Delegate DEFAULT creation to the shared entry point so app + CLI create
    # the Product default identically (single creation path). Returns the df.
    # df is omitted so the sheet is read ONLY when a config is actually created
    # (no Excel read on the common "already exists" path).
    p = cp.load_or_create_default_config(path, sheet, regenerate=regenerate)
    return pd.read_csv(p)


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
                cfg = cp.ModelConfig(
                    target=tgt, model_weeks=int(window),
                    variable_config=resolve_cfg_path(path, sheet, channel))
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


def fmt_km(v, decimals: int = 2) -> str:
    """Chart data-label format (Alex, slide 3): 1,200,000→1.2m · 550,000→550k
    · 10,675→10.68k. Up to `decimals` places, trailing zeros trimmed. Shared
    helper — reused by the export tab later."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    a = abs(v)
    sign = "−" if v < 0 else ""
    if a >= 1e6:
        num, unit = a / 1e6, "m"
    elif a >= 1e3:
        num, unit = a / 1e3, "k"
    else:
        num, unit = a, ""
    s = f"{num:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{sign}{s}{unit}"


def _fmt_vif(v):
    """VIF cell text + color. Forced collinear variables can be huge/∞ by
    design (they bypass VIF pruning) — show that plainly, don't mislead."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—", INK3
    if np.isinf(v) or v >= 1000:
        return ("∞" if np.isinf(v) else "≥1000"), RED
    if v > VIF_WARN:
        return f"{v:.1f}", WARN
    return f"{v:.1f}", INK2


# ---------- Contributions time filter (C1 / E3 / E4) ----------
# The "view period" is decoupled from the modeling WINDOW (E4): the model is fit
# on the window, but the avg-weekly-due-to view can be aggregated over any
# sub-period — entire window, a model year (current/previous), or the latest
# 13/4 weeks — by re-aggregating the stored per-week contributions.
CONTRIB_TAIL_PERIODS = [("q4", "Quarter 4 (Oct–Dec, latest)"),
                        ("last13", "Latest 13 weeks"),
                        ("last4", "Latest 4 weeks")]


def _year_months(cw):
    return [(int(d[:4]), int(d[5:7])) for d in cw.get("dates", []) if len(d) >= 7]


def _period_week_idx(cw, period, time_map=None):
    n = len(cw.get("dates", []))
    if time_map and isinstance(period, str) and period.startswith("map:"):
        wanted = set(time_map.get(period[4:], []))     # uploaded week→period map
        return [i for i, d in enumerate(cw.get("dates", [])) if d in wanted]
    if period == "last13":
        return list(range(max(0, n - 13), n))
    if period == "last4":
        return list(range(max(0, n - 4), n))
    if period == "q4":                          # calendar Q4 of the latest year
        ym = _year_months(cw)
        q4y = [y for (y, m) in ym if m in (10, 11, 12)]
        if not q4y:
            return []
        ty = max(q4y)
        return [i for i, (y, m) in enumerate(ym)
                if y == ty and m in (10, 11, 12)]
    yl = cw.get("year_labels", [])
    # semantic, window-stable: current = latest model year, previous = prior.
    # If the year doesn't exist (e.g. no previous year in a 52-wk window) return
    # EMPTY, never the fallthrough to "all weeks" (which would mislabel).
    if period == "current":
        return [j for j, v in enumerate(cw.get("years", [])) if yl and v == yl[-1]]
    if period == "previous":
        return ([j for j, v in enumerate(cw.get("years", [])) if v == yl[-2]]
                if len(yl) >= 2 else [])
    if period and period.startswith("Y") and period[1:].isdigit():  # legacy
        i = int(period[1:]) - 1
        if 0 <= i < len(yl):
            return [j for j, v in enumerate(cw.get("years", [])) if v == yl[i]]
    return list(range(n))                       # "all" / default


def _period_label(cw, period, time_map=None):
    if isinstance(period, str) and period.startswith("map:"):
        return f"{period[4:]} (mapped)"
    if period == "last13":
        return "Latest 13 weeks"
    if period == "last4":
        return "Latest 4 weeks"
    if period == "q4":
        q4y = [y for (y, m) in _year_months(cw) if m in (10, 11, 12)]
        return f"Q4 {max(q4y)} (Oct–Dec)" if q4y else "Quarter 4"
    yl = cw.get("year_labels", [])
    if period == "current":
        return f"Current year — {yl[-1]}" if yl else "Current year"
    if period == "previous":
        return f"Previous year — {yl[-2]}" if len(yl) >= 2 else "Previous year"
    if period and period.startswith("Y") and period[1:].isdigit():
        i = int(period[1:]) - 1
        return yl[i] if 0 <= i < len(yl) else period
    return "Entire window"


def _avgc_values(cw, period, time_map=None):
    """Avg weekly due-to per driver over the selected period; media averaged
    only over that period's execution weeks (matches the engine's rule)."""
    idx = _period_week_idx(cw, period, time_map)
    media, exec_ = set(cw.get("media", [])), cw.get("exec", {})
    out = {}
    for d in cw.get("drivers", []):
        vals = cw["contrib"][d]
        if d in media and d in exec_:
            ex = exec_[d]
            sel = [vals[i] for i in idx if ex[i] > 0]
        else:
            sel = [vals[i] for i in idx]
        out[d] = (sum(sel) / len(sel)) if sel else 0.0
    return out, len(idx)


def _avgc_figure(cw, period, time_map=None, compare=None):
    if not cw or not cw.get("drivers"):
        return _fig_base(go.Figure(), 320)
    vals, _ = _avgc_values(cw, period, time_map)
    if compare and compare != "none":               # period-vs-period (E4)
        vb, _ = _avgc_values(cw, compare, time_map)
        names = sorted(vals, key=lambda d: vals[d])  # order by primary period
        f = go.Figure()
        f.add_bar(y=names, x=[vals[d] for d in names],
                  name=_period_label(cw, period, time_map), orientation="h",
                  marker_color=NAVY, text=[fmt_km(vals[d]) for d in names],
                  textposition="outside", textfont=dict(size=8),
                  cliponaxis=False)
        f.add_bar(y=names, x=[vb[d] for d in names],
                  name=_period_label(cw, compare, time_map), orientation="h",
                  marker_color="#0E7090", text=[fmt_km(vb[d]) for d in names],
                  textposition="outside", textfont=dict(size=8),
                  cliponaxis=False)
        f.update_layout(barmode="group")
        _fig_base(f, max(340, 44 * len(names)))
        return f
    ser = sorted(vals.items(), key=lambda kv: kv[1])
    names = [k for k, _ in ser]
    xs = [v for _, v in ser]
    f = go.Figure(go.Bar(
        y=names, x=xs, orientation="h",
        marker_color=[("#D9705E" if v < 0 else NAVY) for v in xs],
        text=[fmt_km(v) for v in xs], textposition="outside",
        textfont=dict(size=9), cliponaxis=False))
    _fig_base(f, max(320, 30 * len(names)))
    return f


def _avgc_caption(cw, period, time_map=None, compare=None):
    na = len(_period_week_idx(cw, period, time_map)) if cw else 0
    if compare and compare != "none":
        nb = len(_period_week_idx(cw, compare, time_map))
        return (f"Comparing {_period_label(cw, period, time_map)} ({na} wks) "
                f"vs {_period_label(cw, compare, time_map)} ({nb} wks) · "
                "signed avg weekly (media over execution weeks)")
    return ("Signed; media averaged over execution weeks only · "
            f"{_period_label(cw, period, time_map)} ({na} wks)")


def _compare_options(cw, time_map=None, period=None):
    # exclude the current PRIMARY period so you can't compare A vs A. (Different
    # tokens that happen to cover the same dates — e.g. built-in Q4 vs a mapped
    # Q4 — are still allowed; only the identical token is removed.)
    return [{"label": "No comparison", "value": "none"}] + \
        [o for o in _avgc_period_options(cw, time_map) if o["value"] != period]


def _avgc_period_options(cw, time_map=None):
    # Semantic, window-stable labels (Alex): Current / Previous year always map
    # to the latest / prior model year regardless of the modeling window.
    yl = (cw or {}).get("year_labels", [])
    opts = [{"label": "Entire window", "value": "all"}]
    if yl:
        opts.append({"label": f"Current year — {yl[-1]}", "value": "current"})
    if len(yl) >= 2:
        opts.append({"label": f"Previous year — {yl[-2]}", "value": "previous"})
    opts += [{"label": l, "value": v} for v, l in CONTRIB_TAIL_PERIODS]
    # uploaded mapping-file periods (E3), scoped to weeks present in this run
    if time_map and cw:
        present = set(cw.get("dates", []))
        for lbl, dts in time_map.items():
            if present & set(dts):
                opts.append({"label": f"⤷ {lbl} (mapped)",
                             "value": f"map:{lbl}"})
    return opts


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


def _defn(term, *body):
    return html.Div(style={"marginBottom": "11px"}, children=[
        html.Span(term, style={"fontWeight": 600, "color": INK}),
        html.Span(" — ", style={"color": MUTED}),
        html.Span(list(body), style={"color": INK2}),
    ])


def _defs_section(title, *items):
    return html.Div(className="card card-pad", style={"marginBottom": "14px"},
                    children=[html.Div(title, className="card-title",
                                       style={"marginBottom": "10px"})]
                    + list(items))


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


def pick(label, dd_id, width="140px", wide=False, help=None):
    lab = html.Div(
        [label] + ([html.Span(" ⓘ", style={"color": MUTED,
                                            "cursor": "help"})] if help else []),
        className="plabel", title=help or "")
    return html.Div([
        lab,
        dcc.Dropdown(id=dd_id, clearable=False,
                     style={"width": width}),
    ], className="bench-pick" + (" wide" if wide else ""), title=help or "")


app.layout = html.Div(className="bench-app", children=[
    dcc.Store(id="datapath"),
    dcc.Store(id="batch_sel"),
    # most-recent Diagnostics run — shared context for the Variables tab
    # (coef/due-to columns), and reused later by data-download & VIF views
    dcc.Store(id="last_run"),
    # active workspace tab — driven by the left nav and by Run (which
    # redirects to Diagnostics). Single renderer keeps show/hide in one place.
    dcc.Store(id="active_tab", data="diag"),
    # bumped AFTER any config write/delete (save / regenerate / reset). The
    # table reload and scope-status readers depend on it, so Dash's dependency
    # graph guarantees they re-read disk only after the write — no race.
    dcc.Store(id="config_rev", data=0),
    # per-week contributions of the latest run — lets the Contributions time
    # filter re-aggregate the avg-weekly view without re-running the model.
    dcc.Store(id="contrib_wk"),
    # uploaded week→period mapping (E3): {period label: [YYYY-MM-DD, …]}.
    dcc.Store(id="time_map"),

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
                pick("WINDOW", "window", width="120px",
                     help=("Training window — how many of the most recent "
                           "weeks the reported model is fit on (52 / 104 / 156 "
                           "wks). The last up to 13 weeks are reserved for "
                           "validation (holdout), independent of the window, so "
                           "a holdout normally exists — except a very short "
                           "history (needs ≥5 training wks) or holdout set to "
                           "0. Due-to year splits are taken within the window. "
                           "See the Definitions tab.")),
            ]),
            html.Div(style={"flex": 1}),
            # Run lives on Variables (tuning) and Batch (per-slice) only — the
            # top bar just reports the latest run. (Alex: slides 1–4)
            html.Div(id="runstat", className="bench-runstat"),
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
                html.Div("CONFIG", className="plabel",
                         title="Edit the Product default (all channels) or a "
                               "Channel-specific override. At runtime an "
                               "override wins over the default for that "
                               "channel."),
                dcc.Dropdown(id="config_scope", clearable=False,
                             options=[{"label": "Product default",
                                       "value": "default"}],
                             value="default",
                             style={"width": "210px", "fontSize": "12px"}),
                html.Div("DUE-TO PERIOD", className="plabel",
                         title="Which model period the due-to context column "
                               "shows (Y1 / Y2 / … / Total). Coefficient is "
                               "the same across periods."),
                dcc.Dropdown(id="dueto_period", clearable=False,
                             options=[{"label": "Total", "value": "Total"}],
                             value="Total",
                             style={"width": "110px", "fontSize": "12px"}),
                dcc.Checklist(id="hide_excluded",
                              options=[{"label": " Hide excluded",
                                        "value": "hide"}],
                              value=[], style={"fontSize": "12.5px",
                                               "color": INK2}),
                html.Div(style={"flex": 1}),
                dcc.Upload(id="tmpl_upload", multiple=False, children=html.Button(
                    "Upload template", className="btn btn-ghost",
                    title="Upload a filled template (.xlsx/.csv). Rows with "
                          "channel=ALL write the Product default; a specific "
                          "channel writes that override; variables not listed "
                          "for a group become excluded.")),
                html.Button("Download template", id="tmpl_dl_btn", n_clicks=0,
                            className="btn btn-ghost",
                            title="Download this product's config (default + "
                                  "overrides) as an editable template."),
                dcc.Download(id="tmpl_download"),
                html.Button("Reset to default", id="cfg_reset", n_clicks=0,
                            className="btn btn-ghost",
                            title="In a channel scope, delete this channel's "
                                  "override so it inherits the Product default "
                                  "again."),
                html.Button("Regenerate defaults", id="cfg_regen", n_clicks=0,
                            className="btn btn-ghost"),
                html.Button("Save config", id="cfg_save", n_clicks=0,
                            className="btn btn-ghost"),
                html.Button("▶  Run model", id="run_vars", n_clicks=0,
                            className="btn btn-primary",
                            style={"height": "36px"}),
            ]),
            html.Div(id="cfg_scope_status",
                     style={"fontSize": "12px", "color": NAVY,
                            "fontWeight": 500, "marginTop": "2px"}),
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
                filter_action="native",
                columns=[
                    {"id": "variable", "name": "variable", "editable": False},
                    {"id": "family", "name": "family"},
                    {"id": "sign", "name": "expected sign",
                     "presentation": "dropdown"},
                    {"id": "adstock_decay", "name": "adstock", "type": "numeric"},
                    {"id": "coef_lower", "name": "bound lo", "type": "numeric"},
                    {"id": "coef_upper", "name": "bound hi", "type": "numeric"},
                    {"id": "cur_coef", "name": "coef now", "editable": False},
                    {"id": "cur_dueto", "name": "due-to", "editable": False},
                    {"id": "role", "name": "role", "presentation": "dropdown"},
                ],
                style_cell_conditional=(
                    [{"if": {"column_id": c}, "textAlign": "right",
                      "fontFamily": "IBM Plex Mono, monospace"}
                     for c in ("adstock_decay", "coef_lower", "coef_upper",
                               "cur_coef", "cur_dueto")]
                    + [{"if": {"column_id": c}, "color": INK3,
                        "backgroundColor": "#FBFCFD"}
                       for c in ("cur_coef", "cur_dueto")]),
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
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "10px", "marginBottom": "10px"}, children=[
                html.Div("TIME MAP", className="plabel",
                         title="Optional: upload a .csv/.xlsx with 'date' (or "
                               "'week') + 'period' columns to define custom "
                               "periods (Current Year, Quarter 4, Latest 13 "
                               "weeks, …). They appear in the period filter as "
                               "“⤷ … (mapped)”."),
                dcc.Upload(id="timemap_upload", multiple=False,
                           children=html.Button("Upload time map",
                                                className="btn btn-ghost")),
                html.Div(id="timemap_status",
                         style={"fontSize": "12px", "color": INK3}),
            ]),
            html.Div(style={"display": "flex", "gap": "14px"}, children=[
                html.Div(style={"flex": 1.2}, children=[card(
                    "Due-to totals by model year", None,
                    dcc.Graph(id="fig_cby",
                              config={"displayModeBar": False}))]),
                html.Div(style={"flex": 1}, children=[html.Div(
                    className="card card-pad", children=[
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "justifyContent": "space-between",
                                    "gap": "10px", "marginBottom": "2px"},
                             children=[
                        html.Div("Avg weekly due-to per driver",
                                 className="card-title"),
                        html.Div(style={"display": "flex", "gap": "6px",
                                        "alignItems": "center"}, children=[
                            dcc.Dropdown(id="avgc_period", clearable=False,
                                         options=[{"label": "Entire window",
                                                   "value": "all"}],
                                         value="all",
                                         style={"width": "160px",
                                                "fontSize": "12px"}),
                            html.Span("vs", style={"fontSize": "12px",
                                                   "color": MUTED}),
                            dcc.Dropdown(id="avgc_compare", clearable=False,
                                         options=[{"label": "No comparison",
                                                   "value": "none"}],
                                         value="none",
                                         style={"width": "160px",
                                                "fontSize": "12px"}),
                        ]),
                    ]),
                    html.Div("Signed; media averaged over execution weeks only",
                             id="avgc_sub", className="card-sub",
                             style={"marginBottom": "4px"}),
                    dcc.Graph(id="fig_avgc",
                              config={"displayModeBar": False}),
                ])]),
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
                html.Button("▶  Run model", id="run_batch_one", n_clicks=0,
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
                    page_size=100,
                    style_cell_conditional=[
                        {"if": {"column_id": "grade"},
                         "textAlign": "center", "minWidth": "128px",
                         "width": "128px"}],
                    **TBL_BASE,
                )])]),
        ]),

        # ── Export ──
        html.Div(id="scr-export", className="bench-screen",
                 style={"display": "none"}, children=[
            html.Div("Export", style={"fontSize": "15px", "fontWeight": 600}),
            html.Div("One Excel with three tabs — fit statistics (+ grade), "
                     "fit structure (the variable setup, same format as the "
                     "upload template), and weekly due-tos (actual by date). "
                     "Uses the current TARGET and WINDOW from the top bar and "
                     "the override-wins config.",
                     style={"fontSize": "12px", "color": INK3,
                            "margin": "4px 0 12px"}),
            html.Div(className="card card-pad", children=[
                dcc.RadioItems(
                    id="export_mode",
                    options=[{"label": " This slice (current Product × Channel)",
                              "value": "slice"},
                             {"label": " All Product × Channel combinations",
                              "value": "all"}],
                    value="slice",
                    labelStyle={"display": "block", "margin": "4px 0",
                                "fontSize": "13px"}),
                html.Div(style={"display": "flex", "alignItems": "center",
                                "gap": "12px", "marginTop": "10px"}, children=[
                    html.Button("⤓  Download Excel", id="export_btn",
                                n_clicks=0, className="btn btn-primary",
                                style={"height": "36px"}),
                    html.Div(id="export_status",
                             style={"fontSize": "12px", "color": INK3}),
                ]),
                dcc.Download(id="export_download"),
                html.Div("Note: “All” re-runs every slice and can take a few "
                         "seconds.", style={"fontSize": "11px",
                                            "color": MUTED, "marginTop": "8px"}),
            ]),
        ]),

        # ── Definitions & Methods ──
        html.Div(id="scr-defs", className="bench-screen",
                 style={"display": "none", "fontSize": "13px",
                        "lineHeight": "1.55", "maxWidth": "920px"}, children=[
            html.Div("Definitions & Methods", style={"fontSize": "15px",
                                                     "fontWeight": 600}),
            html.Div("What the numbers mean and how a model is built. "
                     "Definitions track the engine's actual behavior.",
                     style={"fontSize": "12px", "color": INK3,
                            "margin": "4px 0 14px"}),

            _defs_section(
                "Fit metrics",
                _defn("R² / adjusted R²", "share of weekly variance the model "
                      "explains; adjusted R² penalizes adding predictors. Shown "
                      "together so a high R² from over-fitting is visible."),
                _defn("In-sample MAPE", "mean absolute % error on the training "
                      "window, over non-zero-actual weeks only."),
                _defn("Holdout MAPE", "out-of-sample error on a reserved tail. "
                      "The last up to 13 weeks are ALWAYS reserved (independent "
                      "of the window, as long as ≥5 training weeks remain); a "
                      "validation model — same structure, selected without "
                      "seeing the tail — is scored there, and that number is "
                      "kept as the reported holdout."),
                _defn("Durbin–Watson", "residual autocorrelation check (~2 ≈ no "
                      "autocorrelation)."),
                _defn("T-stat", "from an unconstrained OLS as an inference "
                      "reference; |t| ≥ 2 ≈ significant at 5%."),
                _defn("VIF", "variance-inflation factor (collinearity), computed "
                      f"on the final in-model variables. > {int(VIF_WARN)} flags "
                      "redundancy — a non-blocking warning; forced variables can "
                      "be high by design.")),

            _defs_section(
                "Model grade (Model Runs tab)",
                _defn("Good", "holdout MAPE ≤ 10%"),
                _defn("Moderate", f"> 10% and ≤ {int(FLAG_THRESHOLD)}%"),
                _defn("Bad", f"> {int(FLAG_THRESHOLD)}%"),
                _defn("No holdout", "the slice has no holdout period to score."),
                html.Div("These cutoffs are a team convention (reusing the "
                         "review-flag threshold) — confirm with the client.",
                         style={"fontSize": "12px", "color": MUTED,
                                "marginTop": "2px"})),

            _defs_section(
                "Contributions (due-tos)",
                _defn("Due-to (contribution)", "the additive decomposition of "
                      "the fitted line: the intercept as its own line plus "
                      "coefficient × value for each driver, so intercept + "
                      "Σ(βᵢ·xᵢ) reproduces the fitted value exactly each week."),
                _defn("Avg weekly due-to", "the mean of a driver's weekly "
                      "contribution. For ", html.B("media"), " (any variable "
                      "with an adstock decay) the average is taken ", html.B(
                          "only over weeks with real execution"), " (non-zero "
                      "raw spend/impressions) — never diluted by the zero weeks "
                      "of a flight that started mid-window. Non-media drivers "
                      "average over all weeks. (This is the metric Alex asked to "
                      "have defined here.)"),
                _defn("Due-to by model year / YoY", "signed sums per model year "
                      "with the year-over-year change, to see what drove the "
                      "change.")),

            _defs_section(
                "Variables — roles & configuration",
                _defn("Expected sign (guardrail)", "a positive-signed variable "
                      "can never enter with a negative coefficient. Custom "
                      "bounds override the sign."),
                _defn("Role — auto / force / exclude", html.B("auto"), ": the "
                      "model decides (selection). ", html.B("force"), ": "
                      "client-mandated — kept in every run, bypasses selection "
                      "and VIF pruning. ", html.B("exclude"), ": never enters."),
                _defn("Adstock", "applied to ", html.B("any variable with an "
                      "adstock decay set"), " (typically media, but the engine "
                      "doesn't restrict by family): normalized carry-over "
                      "aₜ = (1−decay)·xₜ + decay·aₜ₋₁; the adstocked total ≈ the "
                      "raw total (never inflated). Search ≈ 0.2, TV/CTV ≈ 0.7, "
                      "default 0.5."),
                _defn("Bounds (lo / hi)", "optional hard limits on a "
                      "coefficient; lo must be strictly < hi."),
                _defn("Two-tier config", "a Product default (channel = ALL) "
                      "applies to every channel; a Product × Channel override, "
                      "when present, wins for that channel — for the ", html.B(
                          "Batch, Export and CLI"), ". A ", html.B("Variables-"
                      "tab Run is WYSIWYG"), ": it runs exactly the config "
                      "you're editing (default or override), which can differ "
                      "from what the Batch would resolve.")),

            _defs_section(
                "Window, holdout & method",
                _defn("Window", "how many recent weeks the ", html.B("reported"),
                      " model trains on (52 / 104 / 156). Independently, the "
                      "last up to 13 weeks are reserved for validation (the "
                      "holdout) — so there is normally an out-of-sample metric "
                      "even at a full-history window (exceptions: a very short "
                      "history, which needs ≥5 training weeks, or holdout "
                      "explicitly set to 0). Due-to year splits are taken "
                      "within the window; it changes how much history is used, "
                      "not the data."),
                _defn("How a model is built", "the ", html.B("validation"),
                      " model selects the structure — VIF prune to cut "
                      "collinearity, then forward stepwise — on the pre-tail "
                      "window (no leakage) and is scored on the tail; the ",
                      html.B("reported"), " model refits that ", html.B("same "
                      "structure"), " on the full window via ", html.B("sign/"
                      "box-constrained least-squares"), " (a plain ridge can't "
                      "honor the guardrails). Verified numerically equivalent to "
                      "the sponsor's curve_fit approach.")),
        ]),
    ]),
])


# ---------------- callbacks ----------------

@app.callback(Output("active_tab", "data"),
              [Input(f"nav-{k}", "n_clicks") for k, _ in SCREENS],
              prevent_initial_call=True)
def _nav(*_):
    try:
        trig = ctx.triggered_id
    except Exception:          # outside a live callback (tests)
        trig = None
    return (trig or "nav-diag").replace("nav-", "")


@app.callback(Output("active_tab", "data", allow_duplicate=True),
              Input("run_vars", "n_clicks"), Input("run_batch_one", "n_clicks"),
              prevent_initial_call=True)
def _redirect_after_run(_v, _b):
    # Hitting Run (Variables tuning or a Batch single-slice run) jumps to
    # Diagnostics so you land on the model you just ran. (Alex: slide 2)
    return "diag"


@app.callback(
    [Output(f"scr-{k}", "style") for k, _ in SCREENS]
    + [Output(f"nav-{k}", "className") for k, _ in SCREENS],
    Input("active_tab", "data"))
def _render_tabs(active):
    active = active or "diag"
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


def _fmt_coef(c) -> str:
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return "—"
    a = abs(c)
    if a != 0 and a < 0.01:
        return f"{c:.2e}"
    if a < 1000:
        return f"{c:,.2f}"
    return f"{c:,.0f}"


_CTX_BLANK = {"cur_coef": "—", "cur_dueto": "—"}


def _last_run_ctx_map(last_run, brand, channel, target, period) -> dict:
    """{variable -> {cur_coef, cur_dueto}} for the latest run — current
    coefficient and current due-to for the selected period (Y1 / Y2 / … /
    Total). Shown ONLY when the run matches the current Product × Channel ×
    Target, so switching channel/target blanks the context until you re-run
    (never surface a different slice's numbers). (Alex: slide 2, V1.)"""
    if not last_run or not all(
            str(last_run.get(k)) == str(v) for k, v in
            (("brand", brand), ("channel", channel), ("target", target))):
        return {}
    coefs = last_run.get("coef", {}) or {}
    by_period = last_run.get("dueto_by_period", {}) or {}
    duetos = by_period.get(period) or by_period.get("Total", {}) or {}
    out = {}
    for var in set(coefs) | set(duetos):
        out[var] = {"cur_coef": _fmt_coef(coefs.get(var)),
                    "cur_dueto": fmt_compact(duetos.get(var))}
    return out


def _apply_ctx(rows, ctx_map):
    for row in rows:
        row.update(ctx_map.get(str(row.get("variable")), _CTX_BLANK))
    return rows


@app.callback(Output("dueto_period", "options"),
              Output("dueto_period", "value"),
              Input("last_run", "data"), State("dueto_period", "value"))
def _dueto_period_opts(last_run, current):
    periods = (last_run or {}).get("periods") or ["Total"]
    opts = [{"label": p, "value": p} for p in periods]
    val = (current if current in periods
           else "Total" if "Total" in periods else periods[-1])
    return opts, val


@app.callback(Output("cfg_table", "filter_query"),
              Input("hide_excluded", "value"))
def _hide_excluded(val):
    # native filter only hides rows from view — cfg_table.data (what Save
    # reads) stays complete, so excluded rows are never dropped on save.
    return '{role} != "exclude"' if val and "hide" in val else ""


@app.callback(Output("config_scope", "options"),
              Output("config_scope", "value"),
              Input("channel", "value"), State("config_scope", "value"))
def _scope_opts(channel, current):
    opts = [{"label": "Product default (all channels)", "value": "default"}]
    if channel:
        opts.append({"label": f"This channel — {channel}", "value": "channel"})
    # channel scope is only valid when a channel is selected — otherwise force
    # back to default so Save/Run can't silently write the default (#4).
    val = "channel" if (current == "channel" and channel) else "default"
    return opts, val


@app.callback(Output("cfg_status", "children", allow_duplicate=True),
              Output("config_rev", "data", allow_duplicate=True),
              Input("cfg_reset", "n_clicks"),
              State("config_scope", "value"), State("channel", "value"),
              State("brand", "value"), State("datapath", "data"),
              prevent_initial_call=True)
def _reset_override(_n, scope, channel, brand, dp):
    # Delete this channel's override so it inherits the Product default again.
    # config_rev is bumped ONLY on an actual delete, so the table/status readers
    # re-read disk after the file is gone (no race). (#3)
    if not dp or not brand:
        return "Load a datafile first.", no_update
    if scope != "channel" or not channel:
        return ("Reset to default applies only when editing a channel override.",
                no_update)
    ov = override_cfg_path(dp["path"], brand, channel)
    if not os.path.exists(ov):
        return (f"{brand} × {channel} has no override — already inherits default.",
                no_update)
    try:
        with _CFG_WRITE_LOCK:
            os.remove(ov)
    except OSError as e:
        return f"Could not remove override: {e}", no_update
    return (f"{brand} × {channel} override removed — now inherits the Product "
            "default (and future default changes apply).", time.time())


@app.callback(Output("config_rev", "data", allow_duplicate=True),
              Output("cfg_status", "children", allow_duplicate=True),
              Input("cfg_regen", "n_clicks"),
              State("config_scope", "value"), State("channel", "value"),
              State("brand", "value"), State("datapath", "data"),
              prevent_initial_call=True)
def _regen_cfg(_n, scope, channel, brand, dp):
    # Regenerate defaults into the CURRENT scope's file (never clobber the
    # Product default while editing a channel override), then bump config_rev
    # so the table reloads the freshly written config deterministically. (#1)
    if not dp or not brand:
        return no_update, "Load a datafile first."
    fresh = fresh_default_df(dp["path"], brand)
    target = save_cfg_path(dp["path"], brand, scope, channel)
    with _CFG_WRITE_LOCK:
        fresh.to_csv(target, index=False)
    where = (f"{brand} × {channel} override" if scope == "channel" and channel
             else f"{brand} default (all channels)")
    return time.time(), f"Regenerated {where} from data — {len(fresh)} variables."


@app.callback(Output("cfg_scope_status", "children"),
              Input("config_scope", "value"), Input("channel", "value"),
              Input("brand", "value"), Input("config_rev", "data"),
              Input("last_run", "data"), State("datapath", "data"))
def _scope_status(scope, channel, brand, _rev, _lr, dp):
    if not dp or not brand:
        return ""
    if scope == "channel" and channel:
        if os.path.exists(override_cfg_path(dp["path"], brand, channel)):
            return (f"Editing {brand} × {channel} override — wins over the "
                    "default for this channel at runtime.")
        return (f"Editing {brand} × {channel} — no override yet (seeded from "
                "the Product default). Save or Run creates the override.")
    return (f"Editing {brand} Product default — used by every channel that "
            "has no override.")


@app.callback(Output("cfg_table", "data"),
              Input("brand", "value"), Input("last_run", "data"),
              Input("dueto_period", "value"), Input("channel", "value"),
              Input("target", "value"), Input("config_scope", "value"),
              Input("config_rev", "data"),
              State("datapath", "data"), State("cfg_table", "data"))
def _load_cfg(brand, last_run, period, channel, target, scope, _rev,
              dp, current):
    if not dp or not brand:
        return []
    try:
        trig_ids = {t["prop_id"].split(".")[0] for t in (ctx.triggered or [])}
    except Exception:
        trig_ids = set()
    ctx_map = _last_run_ctx_map(last_run, brand, channel, target, period)
    # Reload the scoped config from disk when the product or config scope
    # changed, when a write happened (config_rev — save / regenerate / reset),
    # on first paint, or (channel-scope) when the channel changed. For
    # period/target/run/last_run (and channel changes while editing the
    # default) keep the on-screen rows so unsaved edits survive; just re-apply
    # the (slice-gated) context columns.
    reload = ("brand" in trig_ids or "config_scope" in trig_ids
              or "config_rev" in trig_ids or not current
              or ("channel" in trig_ids and scope == "channel"))
    if not reload:
        return _apply_ctx(current, ctx_map)
    cfg, _state = load_cfg_for_edit(dp["path"], brand, scope, channel)
    return _apply_ctx(cfg[CFG_COLS].to_dict("records"), ctx_map)


def _validate_cfg_rows(rows):
    """Coerce numerics + check bounds WITHOUT writing. Returns (ok, cfg_df,
    bad_list). lsq_linear needs lo STRICTLY < hi, so `lo >= hi` is rejected."""
    cfg = pd.DataFrame(rows, columns=CFG_COLS)
    for c in ("adstock_decay", "coef_lower", "coef_upper"):
        cfg[c] = pd.to_numeric(cfg[c], errors="coerce")
    bad = cfg[(cfg.coef_lower.notna()) & (cfg.coef_upper.notna())
              & (cfg.coef_lower >= cfg.coef_upper)]
    return (len(bad) == 0), cfg, list(bad.variable)


def _persist_cfg_rows(rows, target_path):
    """Validate + write a config to `target_path` (Product default or a
    Product×Channel override). Returns (ok, cfg_df, bad_list). Shared by Save
    and by a Variables-tab Run (auto-persist so 'modify variables and hit run'
    works — Alex, slide 2). Context/helper columns are dropped by CFG_COLS."""
    ok, cfg, bad = _validate_cfg_rows(rows)
    if not ok:
        return False, cfg, bad
    with _CFG_WRITE_LOCK:            # serialize with template-upload commits
        cfg.to_csv(target_path, index=False)
    return True, cfg, []


@app.callback(Output("cfg_status", "children"), Output("config_rev", "data"),
              Input("cfg_save", "n_clicks"),
              State("cfg_table", "data"), State("brand", "value"),
              State("config_scope", "value"), State("channel", "value"),
              State("datapath", "data"), prevent_initial_call=True)
def _save_cfg(_n, rows, brand, scope, channel, dp):
    if not dp or not brand:
        return "Load a datafile first.", no_update
    target = save_cfg_path(dp["path"], brand, scope, channel)
    ok, cfg, bad = _persist_cfg_rows(rows, target)
    if not ok:
        return f"NOT saved: bound lo must be < bound hi for {bad}", no_update
    where = (f"{brand} × {channel} override" if scope == "channel" and channel
             else f"{brand} default (all channels)")
    return (f"Saved to {where} — {len(cfg)} variables, "
            f"{int((cfg.role == 'force').sum())} forced, "
            f"{int((cfg.role == 'exclude').sum())} excluded. "
            "Next run uses it.", time.time())


@app.callback(Output("tmpl_download", "data"),
              Input("tmpl_dl_btn", "n_clicks"),
              State("brand", "value"), State("datapath", "data"),
              prevent_initial_call=True)
def _download_template(_n, brand, dp):
    if not dp or not brand:
        return no_update
    tdf = build_template_df(dp["path"], brand)

    def _to_xlsx(buf):
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            tdf.to_excel(xw, index=False, sheet_name="variable_template")
    return dcc.send_bytes(_to_xlsx, f"variable_template_{_slug(brand)}.xlsx")


@app.callback(Output("cfg_status", "children", allow_duplicate=True),
              Output("config_rev", "data", allow_duplicate=True),
              Input("tmpl_upload", "contents"),
              State("tmpl_upload", "filename"), State("datapath", "data"),
              prevent_initial_call=True)
def _upload_template(contents, filename, dp):
    if not contents or not dp:
        return no_update, no_update
    import base64
    import io
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        if str(filename).lower().endswith((".xlsx", ".xls")):
            tdf = pd.read_excel(io.BytesIO(raw))
        else:
            tdf = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return f"Could not read template '{filename}': {e}", no_update
    status, results = apply_template_df(tdf, dp["path"])
    summary = "; ".join(f"{p}×{c}: {s}" for p, c, s in results)
    if status == "applied":
        return f"Template applied — '{filename}': {summary}", time.time()
    if status == "rollback_incomplete":
        # disk DID change and couldn't be fully restored — refresh the UI and
        # surface the recovery path (originals kept as .bak).
        return (f"⚠ PARTIAL WRITE — '{filename}': {summary}", time.time())
    return (f"Template REJECTED (nothing written — fix and re-upload) — "
            f"'{filename}': {summary}", no_update)


@app.callback(Output("export_download", "data"),
              Output("export_status", "children"),
              Input("export_btn", "n_clicks"),
              State("export_mode", "value"), State("brand", "value"),
              State("channel", "value"), State("target", "value"),
              State("window", "value"), State("datapath", "data"),
              prevent_initial_call=True)
def _export(_n, mode, brand, channel, target, window, dp):
    if not dp or not target or not window:
        return no_update, "Load data and pick a target + window first."
    if mode == "slice" and (not brand or not channel):
        return no_update, "Pick a Product and Channel for a single-slice export."
    try:
        writer, n_ok, n_err = build_export_workbook(
            dp["path"], target, window, mode, brand, channel)
    except Exception as e:
        return no_update, f"Export failed: {type(e).__name__}: {e}"
    if n_ok == 0 and n_err == 0:
        return no_update, "Nothing to export (no runnable slices)."
    ds = _slug(os.path.splitext(os.path.basename(dp["path"]))[0])
    fname = (f"export_{_slug(brand)}_{_slug(channel)}.xlsx" if mode == "slice"
             else f"export_all_{ds}.xlsx")
    # download even if every slice failed — the workbook carries the errors
    # sheet so the user can see WHY.
    if n_ok == 0:
        status = f"All {n_err} slice(s) failed — see the 'errors' sheet."
    else:
        status = f"Exported {n_ok} slice(s)" + (f" · {n_err} failed" if n_err
                                                else "")
    return dcc.send_bytes(writer, fname), status


def _summary_for(dp):
    if dp and os.path.exists(batch_summary_path(dp["path"])):
        return batch_summary_path(dp["path"])
    if dp and os.path.normpath(dp["path"]) == os.path.normpath(DEFAULT_DATA) \
            and os.path.exists(SUMMARY):
        return SUMMARY
    return None


def _grade_label(m) -> str:
    # Good ≤10% · Moderate ≤40% · Bad >40% holdout MAPE (Alex: slide 4)
    if pd.isna(m):
        return "No holdout"
    if m <= 10:
        return "Good Model"
    if m <= FLAG_THRESHOLD:
        return "Moderate Model"
    return "Bad Model"


def _batch_payload(s):
    s = s.copy()
    s.insert(0, "grade", s["MAPE_holdout_pct"].apply(_grade_label))
    cols = ([{"name": "MODEL", "id": "grade"}]
            + [{"name": c.replace("_", " "), "id": c}
               for c in s.columns if c != "grade"])
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
    # Label + fully-colored cell (green / amber / red); white text for
    # contrast. NaN-holdout rows fall through to a neutral grey.
    q_mod = (f"{{MAPE_holdout_pct}} > 10 && "
             f"{{MAPE_holdout_pct}} <= {FLAG_THRESHOLD}")
    styles = [
        {"if": {"column_id": "grade"},
         "backgroundColor": "#F2F4F7", "color": INK3, "fontWeight": 600},
        {"if": {"column_id": "grade",
                "filter_query": "{MAPE_holdout_pct} <= 10"},
         "backgroundColor": GREEN, "color": "#fff", "fontWeight": 600},
        {"if": {"column_id": "grade", "filter_query": q_mod},
         "backgroundColor": WARN, "color": "#fff", "fontWeight": 600},
        {"if": {"column_id": "grade",
                "filter_query": f"{{MAPE_holdout_pct}} > {FLAG_THRESHOLD}"},
         "backgroundColor": RED, "color": "#fff", "fontWeight": 600},
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


COEF_GRID = "2.6fr 1fr 0.9fr 1.2fr 0.8fr 1.1fr 1.2fr"
YOY_GRID = "2.4fr 1fr 1fr 1fr 0.8fr"


def coef_grid_rows(r, fit):
    head = html.Div([html.Div(x) for x in
                     ["VARIABLE", "FAMILY", "SIGN", "T-STAT",
                      html.Div("VIF", style={"textAlign": "right"}),
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
        vif_v = np.nan if name == "const" else float(fit.vif.get(name, np.nan))
        _vif_cell, _vif_color = _fmt_vif(vif_v)
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
            html.Div(_vif_cell, className="mono",
                     style={"textAlign": "right", "fontSize": "12px",
                            "color": _vif_color,
                            "fontWeight": 600 if _vif_color in (WARN, RED)
                            else 400}),
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
    Output("last_run", "data"),
    Output("config_rev", "data", allow_duplicate=True),
    Output("contrib_wk", "data"), Output("avgc_sub", "children"),
    Input("run_vars", "n_clicks"), Input("run_batch_one", "n_clicks"),
    State("brand", "value"), State("channel", "value"),
    State("target", "value"), State("window", "value"),
    State("datapath", "data"), State("cfg_table", "data"),
    State("config_scope", "value"), State("avgc_period", "value"),
    State("avgc_compare", "value"), State("time_map", "data"),
    prevent_initial_call=True)
def run_model(_v, _b, brand, channel, target, window, dp, cfg_rows, scope,
              avgc_period, avgc_compare, time_map):
    empty = _fig_base(go.Figure(), 260)
    blank = ([], empty, "", empty, empty, [], "", empty, empty, [])
    rev = no_update            # bumped only if a Variables Run writes a config
    missing = [lbl for lbl, v in
               [("datafile (hit Load data in the rail)", dp),
                ("product", brand), ("channel", channel),
                ("target", target), ("window", window)] if not v]
    if missing:
        msg = "Cannot run — missing: " + ", ".join(missing)
        print(f"[run_model] {msg}", flush=True)
        return (*blank, msg, "", no_update, rev, None, no_update)  # clear stale
    path = dp["path"]
    t0 = time.time()
    print(f"[run_model] {brand} x {channel} · target={target} "
          f"window={window}", flush=True)
    try:
        trig = ctx.triggered_id
    except Exception:
        trig = None
    # Variables-tab Run: persist the on-screen edits first so "modify
    # variables and hit run" takes effect (Alex, slide 2), writing to the
    # scoped file (Product default or this channel's override). Invalid bounds
    # (lo >= hi) ABORT with a visible error rather than silently running the
    # old config. A Variables run is WYSIWYG — it runs the file just edited.
    # Batch/other runs use override-wins resolution.
    try:
        # everything that can raise (config read/create, sheet load, fit) is in
        # ONE try, so a bad path / missing sheet routes to the failure return
        # that clears the contributions store — not an unhandled callback error.
        load_or_create_config(path, brand)      # ensure the default exists
        if trig == "run_vars":
            run_cfg_path = save_cfg_path(path, brand, scope, channel)
            if cfg_rows:
                ok, _, bad = _persist_cfg_rows(cfg_rows, run_cfg_path)
                if not ok:
                    return (*blank,
                            f"Not run — bound lo must be < bound hi for: {bad}."
                            " Fix these and re-run; your edits were not saved.",
                            "edits invalid — not saved", no_update, rev,
                            None, no_update)      # clear stale contributions
                # config written — signal readers even if the model then fails,
                # so the scope status reflects the new override.
                rev = time.time()
        else:
            run_cfg_path = resolve_cfg_path(path, brand, channel)
        cfg = cp.ModelConfig(target=target, model_weeks=int(window),
                             variable_config=run_cfg_path)
        r = cp.run_slice("", brand, channel, config=cfg,
                         df=load_sheet(path, brand))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (*blank, f"Model failed: {type(e).__name__}: {e}",
                "run failed — see console", no_update, rev,
                None, no_update)                  # clear stale contributions
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
    # reported model is fit on ALL weeks in the window (M1) -> fitted spans them
    f1.add_scatter(x=df["date"], y=fit.fitted, name="Fitted (all data)",
                   line=dict(color=NAVY, width=1.7, dash="dash"))
    if len(r["pred_te"]):
        f1.add_scatter(x=df["date"].iloc[split:], y=r["pred_te"],
                       name="Holdout forecast (out-of-sample)",
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
    n_highvif = int(sum(1 for c in r["selected"]
                        if not np.isnan(fit.vif.get(c, np.nan))
                        and fit.vif.get(c, 0) > VIF_WARN))
    coef_sub = (f"{len(r['selected'])} selected of {n_cand} candidates · "
                + ("all pass sign checks" if conflicts == 0
                   else f"{conflicts} sign conflicts")
                + (f" · {n_highvif} with VIF>{int(VIF_WARN)} (collinearity, "
                   "non-blocking)" if n_highvif else " · VIF all ≤10"))

    cby = r["contrib_by_year"]
    year_cols = [c for c in cby.columns if not str(c).startswith("YoY")]
    plot_tbl = cby[year_cols].drop("Intercept", errors="ignore")
    f4 = go.Figure()
    palette = ["#C5CEDC", NAVY, "#0E7090"]
    for i, yc in enumerate(year_cols):
        f4.add_bar(y=plot_tbl.index, x=plot_tbl[yc], name=str(yc),
                   orientation="h", marker_color=palette[i % len(palette)],
                   text=[fmt_km(v) for v in plot_tbl[yc]],
                   textposition="outside", textfont=dict(size=9),
                   cliponaxis=False)
    f4.update_layout(barmode="group")
    _fig_base(f4, max(320, 30 * len(plot_tbl)))

    # per-week contributions for the Contributions time filter (C1/E3/E4):
    # avg-weekly view can be re-aggregated over any sub-period without re-running
    wkc = fit.contributions.reset_index(drop=True)
    wk_dates = df["date"].reset_index(drop=True)
    wk_years = cp.assign_model_years(wk_dates)
    wk_drivers = [str(c) for c in wkc.columns if str(c) != "Intercept"]
    wk_media = [c for c in wk_drivers
                if (sp := r["specs_by_name"].get(c)) is not None
                and sp.adstock_decay is not None]
    contrib_wk = {
        "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                  for d in wk_dates],
        "years": [str(v) for v in wk_years.values],
        "year_labels": [str(c) for c in year_cols],
        "drivers": wk_drivers,
        "contrib": {c: [float(x) for x in wkc[c].values] for c in wk_drivers},
        "media": wk_media,
        "exec": {c: [1 if float(x) > 0 else 0 for x in r["X_raw"][c].values]
                 for c in wk_media if c in r["X_raw"]},
    }
    # clamp a leftover period (e.g. "previous" carried over to a 1-year window,
    # or a mapped period absent from this run) to a valid option so it can't
    # briefly render mislabeled.
    valid = {o["value"] for o in _avgc_period_options(contrib_wk, time_map)}
    period = avgc_period if avgc_period in valid else "all"
    compare = (avgc_compare if avgc_compare in (valid | {"none"})
               and avgc_compare != period else "none")   # never A-vs-A
    f5 = _avgc_figure(contrib_wk, period, time_map, compare)
    avgc_sub = _avgc_caption(contrib_wk, period, time_map, compare)

    yoy_children = yoy_grid_rows(cby.round(0))
    fit_sub = f"{brand} × {channel} · {target}"
    runstat = [f"Config saved · last run ",
               html.Span(time.strftime("%H:%M"), className="mono"),
               f" ({dur:.1f}s)"]

    # shared run context — Variables-tab coef + due-to (per period) columns,
    # and reused by the export view. Keyed by variable name. (Alex V1: current
    # coefficient + due-to for Y1 / Y2 / … / Total.)
    coef_map = {str(name): float(fit.coef[name])
                for name in fit.coef.index if name != "const"}
    drivers = [d for d in cby.index if str(d) != "Intercept"]
    dueto_by_period = {}
    for i, yc in enumerate(year_cols):
        dueto_by_period[f"Y{i + 1}"] = {
            str(d): float(cby.loc[d, yc]) for d in drivers}
    dueto_by_period["Total"] = {
        str(d): float(cby.loc[d, year_cols].sum()) for d in drivers}
    periods = [f"Y{i + 1}" for i in range(len(year_cols))] + ["Total"]
    last_run = {
        "brand": brand, "channel": channel, "target": target,
        "window": window, "periods": periods,
        "year_labels": [str(c) for c in year_cols],
        "coef": coef_map, "dueto_by_period": dueto_by_period,
        "stats": {"r2": float(fit.r2), "adj_r2": float(fit.adj_r2),
                  "mape": float(fit.mape),
                  "holdout_mape": None if np.isnan(ho) else float(ho),
                  "durbin_watson": float(fit.meta["durbin_watson"]),
                  "n_selected": len(r["selected"]),
                  "n_forced": len(r["forced"])},
        "ts": time.strftime("%H:%M:%S"),
    }
    return (metrics, f1, fit_sub, f2, f3, coef_children, coef_sub,
            f4, f5, yoy_children, "", runstat, last_run, rev,
            contrib_wk, avgc_sub)


@app.callback(Output("avgc_period", "options"), Output("avgc_period", "value"),
              Input("contrib_wk", "data"), Input("time_map", "data"),
              State("avgc_period", "value"))
def _avgc_period_opts(cw, time_map, current):
    opts = _avgc_period_options(cw, time_map)
    values = {o["value"] for o in opts}
    return opts, (current if current in values else "all")


@app.callback(Output("avgc_compare", "options"), Output("avgc_compare", "value"),
              Input("contrib_wk", "data"), Input("time_map", "data"),
              Input("avgc_period", "value"), State("avgc_compare", "value"))
def _compare_opts(cw, time_map, period, current):
    # depends on the primary period so changing it drops the now-duplicate
    # compare option (and resets the value to "none" if it collided).
    opts = _compare_options(cw, time_map, period)
    values = {o["value"] for o in opts}
    return opts, (current if current in values else "none")


@app.callback(Output("time_map", "data"), Output("timemap_status", "children"),
              Input("timemap_upload", "contents"),
              State("timemap_upload", "filename"), prevent_initial_call=True)
def _timemap_upload(contents, filename):
    # Parse an uploaded week→period map: 'date' (or 'week') + 'period' columns.
    # A date may appear in multiple periods (e.g. Current Year AND Quarter 4).
    if not contents:
        return no_update, no_update
    import base64
    import io
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        df = (pd.read_excel(io.BytesIO(raw))
              if str(filename).lower().endswith((".xlsx", ".xls"))
              else pd.read_csv(io.BytesIO(raw)))
    except Exception as e:
        return no_update, f"Could not read map '{filename}': {e}"
    cols = {str(c).strip().lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("week")
    pcol = cols.get("period") or cols.get("label")
    if not dcol or not pcol:
        return no_update, ("Map needs a 'date' (or 'week') column and a "
                           "'period' column.")
    # Coerce dates; rows with an invalid/empty date are SKIPPED and reported
    # (an unfiltered NaT would poison sorted() with mixed str/NaN — TypeError).
    # format="mixed" parses each cell independently so a CSV mixing YYYY-MM-DD
    # and MM/DD/YYYY doesn't drop the latter (pandas <2.0 falls back to infer).
    try:
        parsed = pd.to_datetime(df[dcol], errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(df[dcol], errors="coerce")
    bad = parsed.isna()
    n_bad = int(bad.sum())
    bad_rows = [int(i) + 2 for i in np.where(bad.values)[0]]   # 1-based + header
    m = {}
    for d, p in zip(parsed[~bad].dt.strftime("%Y-%m-%d"),
                    df[pcol][~bad].astype(str)):
        p = p.strip()
        if p and p.lower() != "nan":
            m.setdefault(p, set()).add(d)
    m = {k: sorted(v) for k, v in m.items()}
    if not m:
        return no_update, (f"No valid (date, period) rows — {n_bad} row(s) had "
                           "an invalid/empty date." if n_bad
                           else "No (date, period) rows found in the map.")
    skipped = (f" · skipped {n_bad} row(s) with invalid/empty date "
               f"(rows {bad_rows[:5]}{'…' if len(bad_rows) > 5 else ''})"
               if n_bad else "")
    return m, (f"Loaded {len(m)} mapped period(s): "
               + ", ".join(list(m)[:6]) + (" …" if len(m) > 6 else "") + skipped)


@app.callback(Output("fig_avgc", "figure", allow_duplicate=True),
              Output("avgc_sub", "children", allow_duplicate=True),
              Input("avgc_period", "value"), Input("avgc_compare", "value"),
              Input("time_map", "data"), State("contrib_wk", "data"),
              prevent_initial_call=True)
def _contrib_period(period, compare, time_map, cw):
    # Re-aggregate the avg-weekly view over the selected period — optionally
    # side-by-side with a COMPARE period (E4: quarter-to-quarter etc.) — without
    # re-running. `time_map` is an INPUT so re-uploading a map with the SAME
    # period names but different dates still redraws. contrib_wk holds the LAST
    # SUCCESSFUL run and is CLEARED on any failed run — the same event that
    # blanks the other Contributions charts — so this can't diverge.
    if not cw:
        return no_update, no_update
    valid = {o["value"] for o in _avgc_period_options(cw, time_map)}
    period = period if period in valid else "all"       # re-clamp (map changes)
    if compare not in (valid | {"none"}) or compare == period:  # never A-vs-A
        compare = "none"
    return (_avgc_figure(cw, period, time_map, compare),
            _avgc_caption(cw, period, time_map, compare))


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

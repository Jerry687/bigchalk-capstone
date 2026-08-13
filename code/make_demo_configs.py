"""
Ship the variable configs that go with the demo dataset.

Two things beyond what `generate_default_config` would produce on its own:

1. `Trend` is set to `exclude` in every product default. A bare trend
   regressor fits the training window happily and then extrapolates off the
   end of it, which is exactly the failure the reserved tail exists to catch.
   The sponsor-facing project excludes it for the same reason.

2. A Product x Channel OVERRIDE for the hero slice sets ACV Weighted
   Distribution back to `auto`. That is what makes the live demo possible:
   the slice opens with distribution missing and a bad holdout, and flipping
   the role to `force` on the Variables tab fixes it in front of the room.
   It also puts the two-tier config resolution (override wins over product
   default) on screen rather than only on a slide.

    python code/make_demo_configs.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone_pipeline as cp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "Demo Dataset for Presentation.xlsx")
HERO_SHEET, HERO_CHANNEL = "Brand Aurora", "Channel Northgate"


def main():
    xl = pd.ExcelFile(DATA)
    sheets = [s for s in xl.sheet_names if s.lower().startswith("brand")]
    os.makedirs(cp.config_dir(), exist_ok=True)

    for sheet in sheets:
        df = pd.read_excel(xl, sheet_name=sheet)
        cfg = cp.generate_default_config(df)          # ACV role = force
        cfg.loc[cfg["variable"] == "Trend", "role"] = "exclude"
        path = cp.default_config_path(DATA, sheet)
        cfg.to_csv(path, index=False)
        print(f"default  {sheet:<20} -> {os.path.basename(path)}")

        if sheet == HERO_SHEET:
            ov = cfg.copy()
            ov.loc[ov["variable"] == "ACV Weighted Distribution", "role"] = "auto"
            op = cp.override_config_path(DATA, sheet, HERO_CHANNEL)
            ov.to_csv(op, index=False)
            print(f"override {sheet} x {HERO_CHANNEL} -> {os.path.basename(op)}")

    resolved = cp.resolve_config_path(DATA, HERO_SHEET, HERO_CHANNEL)
    assert os.path.basename(resolved).endswith(
        f"__ch_{cp._cfg_slug(HERO_CHANNEL)}.csv"), resolved
    print(f"\nhero slice resolves to the override: {os.path.basename(resolved)}")


if __name__ == "__main__":
    main()

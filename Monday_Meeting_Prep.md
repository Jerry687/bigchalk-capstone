# Monday check-in with Alex — July 13, 2026, 11:00–12:00
*Prep sheet. Demo from the dashboard; numbers below are current as of July 6.*

## 30-second status
All of last Monday's review items are implemented and validated. Phase 3 is done
(80 models, median R² 0.89, median holdout MAPE ~10%, 9-second batch). Phase 4
dashboard is functional: loads any datafile, per-variable controls, contributions,
batch view. Ahead of the scope timeline (~3 weeks of scoped work done in ~2.5).

## Demo script (~15 min)
1. **Adstock fix** — show `adstock()` and the totals check (~1.00 per media line).
   Mention per-media decays (Google 0.2, CTV 0.7). *"You asked us to line up raw vs
   adstocked — totals now match."*
2. **Your code vs ours** — `reference_alex_curvefit.py`: curve_fit and lsq_linear
   agree to 3e-7 on the real model. One slide-worthy sentence: same math, our
   packaging is per-variable so column order can't misalign bounds.
3. **The Brand 1 story** — batch overview tab, sort by holdout MAPE. Drill into
   Brand 1 × Channel 1: distribution collapse, VIF dropped ACV, force-include fixed
   it (73% → 32%). This demos force/exclude working end to end.
4. **Datafile flexibility** — load the synthetic ChocoBar/GummyMix workbook live;
   sheets and targets auto-detected, model runs, media picked up. *"No data is
   clean; nothing is hard-coded."* Mention Brand 0210 quirk.
5. **Run all combinations** — one click, ~10s, summary table appears.

## Questions for Alex
1. **Brand 1, Q4 2025:** known delisting/assortment loss? Remaining 32% holdout
   error looks like an event outside the data. If yes → should we build an
   event-dummy/overlay mechanism?
2. **Trade coefficients < 1:** enforce as a hard upper bound for volume-on-promo
   variables, or leave to analyst judgment per model?
3. **Adstock decays:** happy with priors (search 0.2 / CTV 0.7 / default 0.5), or
   worth an Optuna pass to fit them per media line?
4. **Deployment target:** analyst laptops (pip install) or an internal server?
   Anything we should package differently?
5. **Stretch idea (gauge interest):** an LLM "model interpreter" that drafts due-to
   commentary per slice from the model outputs. NDA question first: is sending
   anonymized model *outputs* (not raw data) to a hosted LLM API acceptable?
6. Logistics: in-person session once we're deep in UI — Loop office?

## Numbers to have ready
- 82 slices → 80 models, 2 skipped (not sold); median R² 0.89; median holdout ~10%;
  8 slices > 40% (structural candidates)
- Brand 1 × Ch1: R² 0.83, in-sample 10.0%, holdout 31.8% (was 73% pre-fix)
- Batch runtime 9s (was 90s); per-slice 0.55s; selection math exact (1e-6)
- Equivalence to his curve_fit: 3.1e-07 max relative coefficient difference

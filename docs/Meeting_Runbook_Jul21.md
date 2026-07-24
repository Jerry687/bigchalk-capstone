# Jul 21 Sponsor Meeting — Runbook
*Big Chalk NU Capstone · Tue Jul 21, 3:30–4:30 pm CT · Google Meet · Alex Hathcock, Arko + team*
*Deck: `BigChalk_Capstone_Update_Jul21.pptx` (9 main slides + appendix A1–A6 for deep dives)*

---

## Goal

Show that every July 6 review item is implemented and validated, demo the working web app
end to end, and leave with answers to the four open questions. Alex goes deep on specs —
the appendix slides exist so no question requires hand-waving.

## Three parts, three presenters

- **Part 1 — Recap & review items** (slides 1–3) · **Edison**. Opens the meeting, owns appendix A2 (config schema) and A5 (due-to math).
- **Part 2 — Results & live demo** (slides 4–7 + demo) · **Boqi**. Runs the demo, owns A1 (adstock) and A4 (equivalence). Feifan is backup demo driver.
- **Part 3 — Deliverables & discussion** (slides 8–9) · **Feifan**. Closes, drives the four decisions, owns A3 (selection pipeline) and A6 (performance).

Handoffs are cued in the speaker notes on slides 2, 4, and 8. Whoever owns an appendix
slide answers that deep-dive regardless of whose part it lands in; the presenter jumps
to the slide. Edison notes Alex's answers throughout (he's not presenting in Part 3).

## Timeline

| When | Part | What | Slides |
|---|---|---|---|
| 0:00–0:03 | 1 · Edison | Open, agenda, thank him for the Jul 6 review | 1 |
| 0:03–0:06 | 1 · Edison | Recap: where we stood Jul 6 (fast — he was there) | 2 |
| 0:06–0:14 | 1 · Edison | The five review items → what shipped; equivalence callout | 3 |
| 0:14–0:18 | 2 · Boqi | Brand 1 × Channel 1 force-ACV case study | 4 |
| 0:18–0:22 | 2 · Boqi | Phase 3: 80 models, 9 s batch; his "cool optimizations" question, answered | 5 |
| 0:22–0:25 | 2 · Boqi | Dashboard overview slide, then switch to browser | 6–7 |
| 0:25–0:42 | 2 · Boqi | **Live demo** (script below) | 7 |
| 0:42–0:45 | 3 · Feifan | Documentation & hygiene | 8 |
| 0:45–0:55 | 3 · Feifan | Open questions 1–4 — get decisions, Edison writes them down | 9 |
| 0:55–0:60 | 3 · Feifan | Bench design direction feedback, logistics, buffer | 9 |

If demo runs long, cut question 3 (decays) — the Optuna answer can come by email.

## Live demo script (~17 min)

Pre-conditions: app already running (`python dashboard.py`), browser at `127.0.0.1:8050`,
hard-refreshed (Ctrl+F5); console visible on second monitor to show the wiring audit line.

1. **Boot proof** (30 s) — point at console: `wiring OK: 39 components, 12 callbacks`.
   One line: "the UI audits its own callback graph at startup — failures are loud, never silent."
2. **Run Brand 1 × Channel 1** (3 min) — Diagnostics screen: metric strip, holdout band on
   the fit chart, coefficient grid — t-stat bars, family chips, **FORCED badge on ACV**.
3. **Live control** (3 min) — Variables screen. Use his own example from Jul 6: bump one
   media decay 0.5 → 0.6, rerun, watch fit move. "Any amount of control handed to the
   person at the front end" — his words, this is that.
4. **Contributions** (3 min) — model-year due-tos with YoY pills ("how much better am I
   this year than last year"), point out **negative due-tos** (price), and the
   execution-masked weekly averages (no averaging over zero weeks).
5. **Run all** (3 min) — Batch screen, one click, ~9 s, 80 rows. Health dots, sort by
   holdout MAPE, click a flagged row to drill in.
6. **Alien workbook finale** (4 min) — load the synthetic new-client file (ChocoBar/GummyMix):
   sheets and targets auto-detect, model runs, media picked up. Close with his own bar:
   "the hard part is making sure it works from any start all the way to the end."

**Fallbacks:** if the app dies, restart takes ~5 s (wiring audit reconfirms); if it stays
dead, slides 6 + appendix cover every screen — narrate from the deck and offer a recorded
follow-up. Keep a spare terminal open. Do not debug live for more than 60 seconds.

## Deep-dive map — if Alex probes, jump to the appendix

| If he asks about… | Slide | Key points (all verified in code) |
|---|---|---|
| Adstock equation / totals | **A1** | `a_t = (1−λ)·x_t + λ·a_{t−1}`; worked pulse example at λ=0.8 and 0.1; totals check ~1.00 runs every model; per-media decays (search 0.2, CTV/TV 0.7, default 0.5). Saturation/burn-in deliberately out of scope — his call on Jul 6. |
| Config table / variable mapping | **A2** | Columns: `variable, family, sign, adstock_decay, coef_lower, coef_upper, role`. Blank bounds = unconstrained-but-signed (his exact semantics). Per-dataset + per-product files. Handles renamed variables (his base-price example) — nothing hard-coded. |
| Selection order / thresholds | **A3** | Leakage screen → VIF prune (iterative, threshold 10, exact one-inversion math) → forward stepwise (p < 0.05, configurable — he asked) → bounded fit. Forced variables bypass VIF and stepwise. Method rationale on the slide (LSQ vs Ridge, stepwise vs Lasso, holdout vs k-fold) — he told us to have this written down. |
| curve_fit vs lsq_linear | **A4** | Both scipy TRF; his Q "can bounds be > 0 as a minimum, e.g. 0.1–0.25?" — yes, demoed in config; equivalence 3.1e-07 on the real model; per-variable bound packaging so column order can't misalign arrays (the failure mode his array code risks). |
| Due-to computation | **A5** | Additive: coef × value, intercept its own line; sums by model year + YoY; negative due-tos exist (price); weekly averages masked to execution weeks. Trade: volume-on-promo coef < 1 bound is machinery-ready — policy decision is open question 2. |
| Batch speed / scalability | **A6** | His Jul 6 question "do you have cool optimizations I don't know of?" — yes: VIF = diag of inverse correlation matrix, stepwise p-values via normal equations; identical results to 1e-6; 2.7 → 0.55 s per slice, 90 → 9 s full batch. |
| Normalization | — (verbal) | We follow his stance: don't normalize; coefficients live in original units, read the due-tos. Big macro variables → small coefficients, large due-tos standing in for base volume — expected. |
| ROI | — (verbal) | Post-processing (due-to × price vs spend), per his Jul 6 framing — happy to add in Phase 4+ if wanted. |

## Don't claim (honesty list)

- Saturation / burn-in curves — **not implemented** (he said skip it).
- Trade coef < 1 — **not enforced by default**; bounds exist, awaiting his policy answer.
- Per-slice (Brand × Channel) config overrides — **not built yet**; configs are per product. It's the named next step.
- Event dummies / overlays — **not built**; that's open question 1 (Brand 1 Q4 2025).
- Residual autocorrelation (DW < 1 on several slices) — known; raise only if he asks (open issue 5 in Findings).

## Prep checklist (by 3:15 pm)

- [ ] `git pull`; `python dashboard.py` boots with `wiring OK`; hard-refresh browser
- [ ] `Anonymized Data for Project.xlsx` at repo root; synthetic new-client workbook path confirmed
- [ ] Dry-run demo beats 2–6 once (~5 min)
- [ ] Deck open in presenter view; appendix slide numbers memorized (A1=11 … A6=16)
- [ ] Meet link tested, screen-share permissions OK; console font enlarged
- [ ] Edison has the four open questions ready to record answers

## Asks before closing

1. Decisions on the four open questions (slide 9).
2. Bench design direction — keep or steer?
3. In-person session at the Loop office (he offered on Jul 6) — propose next week.
4. If demo went well: gauge interest in the LLM due-to interpreter stretch idea; NDA check
   on sending anonymized model outputs (not raw data) to a hosted LLM API.

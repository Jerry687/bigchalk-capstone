# Action Items — Capstone Review with Alex & Arco (2026-07-21)

Context: Client review of the automated regression engine + dashboard. Alex was very positive ("this is awesome," "huge," "very impressed"). Team is ahead of schedule with ~5 weeks left, so more work is coming. Alex's two hard asks before next check-in: **send screenshots of every tab** and **run the runtime/scaling test**. Everything else is refinement he wants worked on this week.

---

## P1 — Due this week (Alex asked directly)

1. **Send screenshots of every dashboard tab to Alex.** He'll reply with notes/clarifications and further requests. (Asked 3x.)
2. **Run the runtime/scaling test and email Alex the results.** Time 1 model, 10 models, the full set (~160), then run it twice in one run (~300+) to show whether scale is linear vs. exponential. Boqi noted the engine is already optimized (~O(N)) — Alex still wants the actual numbers to include in the final presentation as scalability evidence.

## P2 — Dashboard changes to work on this week

3. **Add context columns to the variable/bounds edit screen.** When editing bound-low / bound-high / coefficient, show the *current coefficient* from the latest model run and the *current due-tos* (latest year) so the user isn't "flying blind." Two more contextual columns.
4. **Tie bound edits to the most recent model run** so edits reflect current state.
5. **Add a data-download / export button.** One Excel with tabs for the key outputs: coefficients, model-fit statistics, weekly due-tos, and the config (variables + constraints, ad-stocked vs. not, forced vs. not).
6. **Add labels to the color-coded fit table.** Keep the color but add a good / concern / bad label — dye the row or the first label. (Color + label, possibly emoji.)
7. **Verify the target label.** Top says target = "volume sales" but the table reads "dollar sales." Confirm it's a display mislabel vs. wrong data, and fix.
8. **Save-iterations / config versioning.** Let the user save the current setup, run a few test models, compare, and revert to the prior config — via naming or an exportable/re-uploadable config file.
9. **VIF / correlation warning.** Show a warning when two variables are highly correlated (don't block — analyst can ignore). Surface back-end details generally: VIF, T-stats, VIF-pruning info.
10. **UI polish.** Tighten the ~30px gray gap between the actual-vs-fitted chart and the residual-distribution chart. Minor, but Alex flagged it.

## P3 — Lower priority (only if easy)

11. **Average weekly due-tos by year** (year-1 avg weekly vs. year-2 avg weekly), carried through to impressions/support (avg weekly impressions behind that support). Alex: "If it's easy, cool. If it's hours, don't worry about it." Note he wasn't sure spend was in the data provided.

## P4 — For the final presentation

12. **The Q4-2025 investigation story** (brand1×channel1): ~10% error in-sample but a 73% miss on the last 13 weeks; distributions collapsed and VIF pruning dropped every distribution variable; fixed by forcing ACV weighted distribution in. Alex loved this "fine-tuned looking into cases" narrative — feature it. (Note: per client NDA, don't speculate publicly on the *cause* outside the data.)
13. **Scalability findings** from the runtime test (item 2).
14. **Flagged slices:** ~9 slices with holdout MAPE >40% — present them as flagged, with the next-step framing that a bad fit signals something missing in the data (likely distribution, price, or seasonality per Arco), not necessarily a bad model.

## On the horizon — Alex/Arco to spec (likely next week)

- **Saturation curves on spend data.** Arco raised it; Alex wants to discuss next week ("we can do a saturation on spends").
- **VIF reduction via differential transforms:** apply a long decay to one correlated media var and a short decay to the other to change its shape and cut correlation (Arco's tip).
- **More work TBD** — Alex said he'll "cleanly spec it out" once the team finishes this batch.

## Team's own track (independent of Alex)

- Continue **Phase 1 validation framework**: freeze/reproduce the AD baseline, classify slices (full / limited / unscorable), rolling-origin validation, robust metrics (MASE, WAPE, MAE, forecast bias, residual diagnostics, coefficient stability), and green/yellow/red/unscorable grading.
- The **6 slices with no MAPE value** — Alex said not to worry (some brands simply aren't sold in some channels), but the team flagged wanting to handle them.

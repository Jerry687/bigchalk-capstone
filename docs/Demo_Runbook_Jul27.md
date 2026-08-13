# Demo Runbook — Alex check-in, Mon 2026-07-27, 11:00–12:00

**Goal of this meeting:** prove that every callout in `Dashboard Edits.pptx` landed, and close out the runtime/scaling ask. Then let Alex spec the next batch.

**Format:** live screen-share of the running dashboard, walked in **his slide order** — so he's watching his own annotations get answered one at a time. Do not walk it tab-by-tab in your own order; the whole point is that he recognises his own list.

---

## T-30 — Pre-flight (do this before the call)

```bash
cd Capstone/code
python dashboard.py            # UI at http://127.0.0.1:8050
```

Checklist:

- [ ] App boots clean, no console errors. (Verified importable on 2026-07-27.)
- [ ] **Load data** box points at `Anonymized Data for Project.xlsx`, then hit Load — do this *before* the call so the first thing Alex sees is a loaded app, not a spinner.
- [ ] **Pre-run Brand 1 × Channel 1** at WINDOW 156 so Diagnostics is already populated. A cold Run at minute one is the most common way a demo stalls.
- [ ] Browser zoom ~90%, window maximised, close other tabs. Hide your bookmarks bar.
- [ ] Have `docs/Alex_Dashboard_Edits_Comprehensive_Test_Report.pdf` open in a second tab as the leave-behind.
- [ ] **Re-run the scaling test on your own laptop** so the numbers you quote are your hardware, not a container: `python scaling_test.py` (writes `outputs/scaling_test_summary.csv` + `scaling_test.png`). Have the PNG open in a tab.
- [ ] Silence notifications.

**Fallback if the app misbehaves:** you already sent him screenshots of every tab — pivot to those plus the test-report PDF and keep talking. Don't debug live.

---

## 0. Opening — 60 seconds, no screen share yet

> "Since your PowerPoint we've built every callout on all six slides — the run restructure, VIF, the grade labels, the variable context columns, the template up/download, the Export tab, the Definitions tab, the time filters, and the modelling change to fit on all data. I also ran the scaling test you asked for. I'd like to walk your slides in order so you can check them off, then hear what you want next."

That framing does two things: it tells him the list is *closed*, and it invites the next spec — which is what he said he'd do once this batch was finished.

---

## 1. Slide 1 — Modelling + Diagnostics (≈10 min)

**Start on the Diagnostics tab**, already populated from the pre-run.

**M1 — final model uses all the data.** This is the biggest conceptual change; lead with it.

- Point at the actual-vs-fitted chart and the reserved tail.
- Talk track: *"You asked that the final model use all the data including the holdout. It now does — but we kept a real out-of-sample number instead of relabelling the in-sample fit. Two models per slice: a validation model picks the structure on the window ending before an always-reserved tail of up to 13 weeks and is scored on that tail — that's the holdout MAPE you see. The reported model refits that same structure on all the data — those are the coefficients and due-tos on screen. So the holdout validates the process out-of-sample, and nothing you're looking at was trained blind to recent weeks."*
- Pre-empt the obvious question: *"The reserved tail is always there, so even a full 156-week window gives a real holdout number instead of blank."*

**D1 — VIF column.** Scroll to the coefficient table.

- *"VIF is now a column here. Amber over 10, red at 1000 or infinite, and the subtitle counts high-VIF variables — it flags, it doesn't block, so the analyst can overrule it."*
- Mention T-stats are in the same table (his "surface the back-end details" ask).

**D2 — Avg Wkly Due-To definition.** Don't explain it here — say *"the formula is written out on the Definitions tab, I'll show you at the end"* and move on. Saves four minutes.

**UI polish (item 10).** The gap between the actual-vs-fitted and residual panels is now 14px (was the ~30px he flagged). Worth one clause — *"and we tightened that gap you circled"* — because noticing his cosmetic notes buys goodwill on the substantive ones. Don't spend more than a sentence.

---

## 2. Slide 2 — Variables (≈12 min, this is the demo's centrepiece)

Switch to **Variables**. This tab has the most of his asks and it's the one that *feels* like a product.

**V1 — context columns.** *"You said the analyst was flying blind when editing bounds. Every row now shows the current coefficient and the current due-to from the latest run, with a Y1 / Y2 / Total toggle."*

- **Flip the DUE-TO PERIOD toggle live.** Watching numbers change is worth more than describing it.
- Then the honesty point he'll respect: *"The context is gated to the exact Product × Channel × Target that was run. Switch channel and it blanks rather than showing you a stale number from a different slice."*

**V4 — Run on this tab, edits auto-persist.** Do this live:

1. Change one bound (e.g. tighten a media variable's upper bound).
2. Hit **Run model** — *no separate Save*.
3. Land on Diagnostics for the same slice (that's G4, his redirect ask) and show the coefficient moved.

> This single 30-second sequence demonstrates V4, G2, G4 and the auto-persist fix at once. It's the strongest moment in the demo — don't rush it.

Also worth one line: *"If you type invalid bounds it aborts the run with a visible error rather than silently falling back to the old config."*

**V2 — template download / upload.** Click **Download template**, open the file so he sees the columns.

- *"Same format the Export tab emits, so export → edit in Excel → re-upload round-trips. A variable you leave off a group's rows is treated as excluded — that's the rule you asked for."*
- *"Uploads are all-or-nothing: the whole file validates first, and a failed write rolls back."* He'll like this; he asked for save/revert (item 8) and this is how it's delivered.

**Two-tier config.** Show the CONFIG scope selector.

- *"Product default plus an optional Product × Channel override, and the override wins. Dashboard, batch runner and CLI all resolve it through the same function, so all three agree."*
- Flag it as **your team's design decision, not his ask** — he responds well to that distinction.

---

## 3. Slide 3 — Contributions (≈8 min)

**C2 / C3 — data labels.** Point at the k/M labels. One sentence: *"1.2m, 550k, formatted like you drew it."* Don't dwell.

**C1 / E3 / E4 — time filtering.** This is the substantive one.

- Change the period filter on the **avg weekly due-to** chart. Emphasise: *"This re-aggregates the stored per-week contributions — it does not re-run the model. Instant."*
- Show the built-in periods (entire window, current/previous year, Q4, latest 13 / 4 weeks).
- **Upload a time-map file** if you have one ready — a `date` + `period` CSV. *"If your reporting calendar doesn't match ours, you upload it. A date can belong to more than one period, and mapped periods only appear when they overlap the run."*
- **Period-vs-period comparison:** overlay two periods side by side. *"Quarter to quarter, or year to year, without re-running anything."*

**C4** — mention in passing that at a 156-week window the data spans three model years and everything (toggle, YoY table, charts) renders dynamically for N years.

---

## 4. Slide 4 — Model Runs (≈5 min)

Switch to **Model Runs** (say the rename out loud: *"'Batch' is now 'Model Runs' as you asked"*).

**B1 — grade labels.** *"The health dot is now a labelled, colour-filled cell: Good / Moderate / Bad Model."*

> **Ask him here, don't assume:** *"We set the cut-offs at 10% and 40% holdout MAPE reusing our existing flag threshold, and added a fourth 'No holdout' state. Those numbers are ours, not yours — do you want different ones?"*
>
> This is one of the few open decisions, and asking it inside the demo is more natural than saving it for the end.

**B3 / G3** — "Run all combinations" now sits next to "Run model."

**B4** — hover the WINDOW control to show the tooltip.

**Flagged slices (P4 item 14).** If the batch table is populated, scroll to the bottom: *"About 9 slices come in over 40% holdout MAPE. We're presenting those as flagged rather than hiding them — a bad fit is a signal that something's missing from the data, most likely distribution, price or seasonality, per Arco's point."*

---

## 5. Slides 5–6 — Export + Definitions (≈6 min)

**Export tab (X1/E2).** Click **Export**, run it for all Product × Channel, open the workbook.

- Three sheets: **fit_statistics** (with the grade colour-filled in Excel), **fit_structure** (= the upload template, so it round-trips), **weekly_due_tos** (tidy, per-week, by date).
- *"You asked for one download with coefficients, fit stats, weekly due-tos and the config including adstock and forced flags. That's this."*

**Definitions tab (E1).** Scroll it slowly — this is where you answer D2 and B4 properly.

- *"Fit metrics, the model grade thresholds, how avg weekly due-to is computed, variable roles, adstock, the two-tier config rule, and the window/reserved-tail method. The thresholds here are read from the code, so the doc can't drift from the app."*

---

## 6. The scaling test — his second hard ask (≈5 min)

Share `outputs/scaling_test.png`.

> **Quote your own laptop's numbers**, not the container's. Container reference run (2 CPUs, BLAS pinned to 2 threads, 2026-07-27):

| Scale point | Models | Total | Per model |
|---|---|---|---|
| 1 model | 1 | 0.07 s | 0.071 s |
| 10 models | 10 | 0.59 s | 0.059 s |
| Full set | 80 | 4.54 s | 0.057 s |
| Full set ×2 | 160 | 9.09 s | 0.057 s |

- Linear fit `total = 0.0567 × n + 0.02`, **R² = 1.0000**.
- Doubling the batch doubled the time — **2.00× for 2.00× the models**. No super-linear term.
- One-time workbook read (all 10 brand sheets, 13 MB xlsx): **4.9 s** — paid once per run, not per model.

Talk track: *"It's linear, and the marginal cost of one more model is about six hundredths of a second. Doubling the set doubled the time exactly. The fixed cost is reading the workbook, not the modelling. At this rate ten times the slices — 800 models — is under a minute of engine time."*

**One correction to give him up front:** *"The full set is 80 modelable slices, not the ~160 we said in the meeting. 82 Brand × Channel combinations exist; 2 have an all-zero target because the brand isn't sold in that channel, so they're skipped. The 160 row above is the full set run twice, which is the doubling test you asked for."*

Then: *"The harness is checked into the repo as `code/scaling_test.py`, so you can re-run it on your hardware or on a bigger client file."*

---

## 7. Close — 3 minutes

1. **Confirm the list is closed.** *"That's every callout from all six slides. The item-by-item tracker is in the repo if you want to audit it."*
2. **The one decision you need:** grade thresholds (10 / 40) — if he didn't answer in section 4, ask now.
3. **Invite the next batch.** *"You mentioned saturation curves on spend, and Arco's idea of using differential decay lengths to break correlation between two media variables. We have about four weeks before the final presentation on Aug 25. What do you want prioritised?"*

**Do not raise in this meeting** (they're your team's track, not his): the Phase 1 validation framework, MASE/WAPE metrics, and the 6 no-MAPE slices he already told you not to worry about. Bringing them up dilutes a meeting whose job is to show a finished list.

---

## Timing

| Segment | Minutes |
|---|---|
| Opening | 1 |
| Slide 1 — Diagnostics / M1 / VIF | 10 |
| Slide 2 — Variables | 12 |
| Slide 3 — Contributions | 8 |
| Slide 4 — Model Runs | 5 |
| Slides 5–6 — Export / Definitions | 6 |
| Scaling test | 5 |
| Close + his next spec | 3 |
| **Total** | **50** — leaves 10 min of slack, which you will use |

If you fall behind, cut in this order: C4 (3 model years), B4 tooltip, the time-map upload. Never cut the Variables edit→Run→Diagnostics sequence or the scaling numbers.

---

## Facts to have in your pocket

- Sponsor equivalence still passes: **1.41e-08** max relative coefficient difference vs. Alex's `curve_fit` reference (re-verified 2026-07-27).
- Dashboard: 6 tabs, ~66 components, 27 callbacks.
- 80 modelable slices out of 82 Brand × Channel combinations.
- Q4-2025 story (Brand 1 × Channel 1): ~10% in-sample error, 73% miss on the last 13 weeks → distributions collapsed, VIF pruning dropped every distribution variable → fixed by forcing ACV weighted distribution in, holdout MAPE 73% → 32%. He loved this narrative — save it for the final presentation unless he asks.

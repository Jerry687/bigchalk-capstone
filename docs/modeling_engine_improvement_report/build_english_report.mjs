import fs from "node:fs";
import path from "node:path";

const baseDir = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (m) => m.slice(1)));
const sourcePath = path.join(baseDir, "artifact.json");
const outputPath = path.join(baseDir, "artifact_en.json");
const artifact = JSON.parse(fs.readFileSync(sourcePath, "utf8"));

artifact.manifest.title = "MMM Modeling Engine Improvement Proposal (Approval Draft)";
artifact.manifest.description = "A phased algorithm improvement, validation, and approval proposal based on the current 80-model baseline and the deep-research findings.";

const cards = Object.fromEntries(artifact.manifest.cards.map((card) => [card.id, card]));
cards.models_completed.description = "Number of Brand × Channel models successfully represented in the current authoritative portfolio summary.";
cards.models_completed.metrics[0].label = "Successful models";
cards.median_r2.description = "Median in-sample R² across 80 models; this measures historical fit, not causal identification.";
cards.median_r2.metrics[0].label = "Median R²";
cards.median_holdout_mape.description = "Median single-holdout MAPE among models with a calculable value.";
cards.median_holdout_mape.metrics[0].label = "Median holdout MAPE (%)";
cards.high_error_models.description = "Number of long-tail models with holdout MAPE above 40%.";
cards.high_error_models.metrics[0].label = "Holdout MAPE > 40%";
cards.unscorable_models.description = "Models without a valid holdout MAPE because they have no holdout weeks or all-zero actuals in the holdout period.";
cards.unscorable_models.metrics[0].label = "Currently unscorable models";

const chart = artifact.manifest.charts.find((item) => item.id === "validation_status_chart");
chart.title = "Current Model Validation Status";
chart.subtitle = "80 models, classified by whether holdout MAPE is calculable and whether it exceeds 40%.";
chart.encodings.x.label = "Validation status";
chart.encodings.y.label = "Number of models";
chart.encodings.tooltip[0].label = "Models";
chart.encodings.tooltip[1].label = "Definition";

const tables = Object.fromEntries(artifact.manifest.tables.map((table) => [table.id, table]));

Object.assign(tables.architecture_options_table, {
  title: "Candidate Architectures and Recommended Roles",
  subtitle: "Compared against the current 80-model batch workflow, interpretability requirements, and local Windows runtime constraints."
});
tables.architecture_options_table.columns.forEach((column, index) => {
  column.label = ["Priority", "Option", "Recommended role", "Primary benefit", "Primary risk", "Proposal decision"][index];
});

Object.assign(tables.roadmap_table, {
  title: "Phased Implementation Roadmap",
  subtitle: "Each phase has an independent approval and acceptance gate; approving this report does not authorize every phase."
});
tables.roadmap_table.columns.forEach((column, index) => {
  column.label = ["Order", "Phase", "Scope", "Primary deliverable", "Gate to next phase", "Current approval recommendation"][index];
});

Object.assign(tables.quality_framework_table, {
  title: "Candidate Model Quality Tiers",
  subtitle: "First determine whether a model is eligible to be scored; then classify it as Green, Yellow, or Red. Numeric thresholds require Phase 1 calibration."
});
tables.quality_framework_table.columns.forEach((column, index) => {
  column.label = ["Order", "Tier", "Meaning", "Candidate rule", "Permitted use", "System action"][index];
});

Object.assign(tables.file_plan_table, {
  title: "Expected Code Impact",
  subtitle: "This is an implementation map; it does not mean these files have been changed."
});
tables.file_plan_table.columns.forEach((column, index) => {
  column.label = ["Order", "Code surface", "Planned change", "Compatibility requirement", "Phase"][index];
});

Object.assign(tables.risk_register_table, {
  title: "Risks and Controls",
  subtitle: "Controls focus on false out-of-sample improvement, attribution drift, and uncontrolled engineering scope."
});
tables.risk_register_table.columns.forEach((column, index) => {
  column.label = ["Priority", "Risk", "Impact", "Control", "Stop condition"][index];
});

Object.assign(tables.approval_table, {
  title: "Decisions Required Before Approval",
  subtitle: "The recommendation is to approve Phases 0–1 only; later phases require new approval supported by experiment results."
});
tables.approval_table.columns.forEach((column, index) => {
  column.label = ["Order", "Decision", "Recommended choice", "Rationale"][index];
});

const blocks = Object.fromEntries(artifact.manifest.blocks.map((block) => [block.id, block]));
blocks.title.body = "# MMM Modeling Engine Improvement Proposal (Approval Draft)";
blocks.technical_summary.body = `## Technical conclusion: preserve the system shell and replace the validation and estimation core in stages

The current engine is already a functioning, interpretable, and operationally stable MMM baseline. It should not be rewritten from scratch. The appropriate target is to preserve Excel ingestion, variable configuration, CSV outputs, the Due-to structure, and Dashboard-facing interfaces; first establish credible time-series validation and model grading; then replace the Stepwise + VIF primary path with constrained regularization; and only afterward evaluate bounded adstock/saturation and Bayesian or hierarchical pilots.

**This proposal recommends approving only Phases 0–1 now.** Phase 2 and every later phase must return for approval with champion–challenger evidence from identical time splits. This fixes the more fundamental problem—our current inability to judge model quality consistently—before adding algorithmic complexity.`;

blocks.baseline_interpretation.body = `## The baseline is promising, but long-tail and unscorable models prevent portfolio-wide approval

The authoritative summary contains 80 successful models, with median R² of 0.88505 and median holdout MAPE of 9.91%. These results show that the current engine captures much of the historical variation in most slices, but they do not by themselves establish generalization or causal attribution: 9 models have holdout MAPE above 40%, and another 6 do not have a valid holdout MAPE.

The success criterion for the improvement program therefore cannot be “higher average R².” It must jointly improve out-of-sample error, tail performance, scoreability, residual structure, and contribution stability.`;

blocks.scope_definitions.body = `## The improvement scope is modeling credibility—not UI expansion or stronger causal claims

The **baseline model** is the current pipeline in \`capstone_pipeline.py\`: geometric adstock, VIF pruning, forward stepwise selection, SciPy constrained least squares, and relative-contribution dead-variable pruning. The **champion** is the production candidate. A **challenger** is an alternative algorithm compared fairly on identical inputs and outer time folds. **Model-implied contribution** is the coefficient multiplied by the transformed variable; it is not the same as experimentally identified incremental sales.

This report does not include a Dashboard redesign, browser file upload, deployment, a budget optimizer, or a full Bayesian rewrite. It also does not change the meaning of the current configuration, summary, or Due-to CSV schemas. Any future fields must be appended compatibly and checked across every consumer.`;

blocks.architecture_recommendation.body = `## Recommended hybrid architecture: constrained regularization by default, the current model as fallback, and Bayesian methods only for targeted upgrades

The research report's central diagnosis is sound: the primary weakness is not linear modeling itself, but weak validation, unstable variable selection, and excessive nonlinear freedom for slices with roughly 104 weeks of data. The default champion should gradually become “bounded media transformations + constrained Ridge/Elastic Net + nested time-series validation + stability checks.” The current constrained least-squares model should remain as a reproducible baseline and fallback.

Bayesian or hierarchical MMM should be piloted only for short-history, high-value, extreme-coefficient, or experimentally calibrated slices—not used as the first full-portfolio replacement.`;

blocks.target_algorithm.body = `## The target algorithm closes the loop from eligibility through transformation, fitting, validation, and grading

1. **Data eligibility:** Evaluate history length, nonzero weeks, target variation, and feasible outer folds; otherwise assign \`Unscorable\` or a pooled-only route.
2. **Feature policy:** Classify variables as Price, Distribution, Trade, Media, Competition, Macro, Seasonality, or Event, then apply one leakage blacklist and consistent sign/forced-variable rules.
3. **Temporal baseline:** Select trend and annual Fourier terms from a small candidate grid; explain time structure before attributing remaining variation to media.
4. **Media transformations:** Permit only a predefined, compact adstock/saturation library, with all choices made exclusively inside inner folds.
5. **Constrained regularization:** Standardize continuous variables; use constrained Ridge as the default candidate and light Elastic Net as a challenger; do not penalize forced variables.
6. **Stability and pruning:** Replace a single relative-contribution threshold with absolute contribution, relative contribution, fold stability, and outer-error impact.
7. **Diagnostics:** Output WAPE, MASE, MAE, bias, Durbin–Watson, Ljung–Box, coefficient-sign stability, and contribution stability.
8. **Quality gate:** Determine scoreability first, then assign Green, Yellow, or Red; low-grade models cannot be approved for budget interpretation.`;

blocks.roadmap_intro.body = `## The roadmap advances through small, reversible stages with a new approval at every gate

Phase 0 establishes tests and freezes the baseline. Phase 1 adds eligibility, time validation, metrics, and grading without changing the regression algorithm. Only after those results are stable should the project test temporal baselines and regularized estimators. This separates changes to the evaluation system from changes to the algorithm itself, so result movements remain explainable.`;

blocks.phase_0_1_detail.body = `## Recommended first authorization: Phases 0–1, with no change to current coefficients

### Phase 0: establish an immutable baseline

- Add automated tests for date parsing, leakage exclusion, adstock, sign/bound constraints, contribution reconstruction, and the current 80-model summary.
- Freeze the Brand × Channel slice inventory, input-data fingerprint, and current output baseline.
- Add deterministic rolling-origin splitter tests, but do not connect them to current production outputs yet.
- Write all validation artifacts to a separate test or experiment directory; never overwrite existing files in \`outputs/\`.

### Phase 1: change evaluation only, not the estimation core

- Introduce Full, Limited, and Reject/pooled-only data eligibility states.
- Add rolling-origin or expanding-window outer folds and MASE, WAPE, MAE, and bias.
- Add \`Unscorable\`; never allow holdout=0 or an all-zero denominator to pass by default.
- Produce a baseline portfolio quality distribution and threshold-sensitivity report.

**At the end of Phase 1, existing coefficients and Due-to results should remain unchanged; only validation evidence and model status should change.**`;

blocks.quality_framework_intro.body = `## Model grades use hard gates—not an arbitrary weighted score whose components can cancel one another

The Green/Yellow/Red/Unscorable structure proposed by the research report is appropriate. However, 91 weeks, 40 nonzero weeks, MASE 0.90, WAPE 15%, and similar values are project-level candidate thresholds, not validated industry laws. Phase 1 must show how changes in these thresholds affect the classification of all 80 models before the project owner approves final rules.

Before those thresholds are approved, the system may calculate candidate grades, but it must not use them as automated business-release rules.`;

blocks.validation_design.body = `## A new algorithm may replace the champion only if it wins consistently on identical outer time folds

The candidate design for Full-eligible slices is three expanding-window outer folds, each with a 13-week test period and up to 104 training weeks. Inner folds are used only to select Fourier order, media transformations, and regularization parameters. Limited slices receive one or two feasible folds and are capped at Yellow. Scaling, transformation, selection, and pruning must learn exclusively from training data.

Acceptance evidence spans five areas: outer MASE/WAPE/MAE/bias, residual structure, coefficient-sign stability, major-contribution stability, and business constraints. Portfolio reporting must show the median, 75th/90th percentile tail, Red/Unscorable counts, and paired differences by slice. The research report's proposed “10% median MASE improvement” should remain a candidate threshold until Phase 1 establishes and calibrates the baseline.`;

blocks.file_plan_intro.body = `## Code changes should extend the existing single engine, not create a second set of business logic

Every consumer should continue to call one compatible \`run_slice\` entry point. The splitter, metrics, quality gate, and challenger estimator should be separate testable modules, but the Dashboard, single-model runner, and batch runner must not maintain different algorithms. New output fields must be appended compatibly while existing fields remain available during migration.`;

blocks.dependency_policy.body = `## Dependency policy: no heavy modeling libraries in Phases 0–1; evaluate CVXPY and PyMC only at later gates

Phases 0–1 require only a test framework and capabilities already available in NumPy, pandas, SciPy, and statsmodels. If the constrained Ridge/Elastic Net prototype proves worthwhile in Phase 3, evaluate \`cvxpy\` and OSQP/Clarabel separately for installation, solver stability, Windows compatibility, and batch performance. Do not add them to production merely because the research report recommends them. PyMC, ArviZ, and PyMC-Marketing are Phase 4 pilot dependencies only.

Before any new dependency is approved, pin the current Python environment, establish reproducibility, and preserve a fallback path without the heavy dependency.`;

blocks.risk_intro.body = `## The largest risk is not insufficient improvement—it is false out-of-sample improvement or an unexplained change in attribution

If any tuning, scaling, transformation selection, or pruning step sees the outer fold, the challenger's result is invalid. A second major risk is that adding a time baseline reallocates Due-to away from media. That may be a more accurate attribution, or it may be competition among model degrees of freedom. It must be explained through fold stability and business review, not prediction error alone.`;

blocks.non_goals.body = `## Explicitly out of scope for this approval

- Do not modify \`capstone_pipeline.py\`, configuration CSVs, the Dashboard, or any existing model output before approval.
- Do not switch the full portfolio directly to Bayesian MMM.
- Do not freely optimize adstock and saturation continuously for every slice.
- Do not treat higher R² as the primary success criterion.
- Do not relabel model-implied contribution as true causal incrementality.
- Do not use the Robyn Python Beta as a production runtime.
- Do not combine UI upload, export, cache, and deployment changes with this modeling effort; those belong to separate engineering workstreams.`;

blocks.approval_intro.body = `## Recommended approval model: authorize validation infrastructure first, then approve algorithm replacement only from evidence

The current recommendation is to authorize Phases 0–1 only and require a read-only baseline validation report. Phases 2–3 must present paired results for the current engine and challenger on exactly the same outer folds. Phase 4 requires a separately approved pilot-slice list and compute budget.`;

blocks.limitations_questions.body = `## Uncertainty and open questions

This approval draft combines the user-provided deep research, current code inspection, and the existing batch results. It did not re-resolve the temporary citation handles embedded in the research document, so external literature claims must be converted into traceable formal references before Phase 3 or Phase 4. Quality thresholds have not yet undergone sensitivity calibration across all 80 slices. The project also lacks experiment or geo-level identification data needed for strictly causal ROI claims.

Before implementation, confirm the sponsor's decisions on event variables, Trade coefficient bounds, adstock optimization, and deployment; which variables are media and may use saturation; whether the primary business objective is prediction, explanation, or budget direction; whether short-history slices may use a pooled-only route; and whether new quality fields may be appended to the current summary CSV.`;

blocks.final_recommendation.body = `## Final recommendation

Approve an **incremental core upgrade**, not a full rewrite. The first implementation step should be Phase 0: freeze a reproducible baseline and add automated tests. Phase 1 should then add time-series validation, robust error metrics, data eligibility, and \`Unscorable\` status—without changing current coefficients or Due-to calculations. Only after reviewing the resulting 80-model quality distribution should the project decide whether to proceed to temporal baselines and a constrained-regularization challenger.

This route preserves the engineering assets that already work while ensuring that future algorithm changes are falsifiable, reversible, and fairly comparable.`;

const data = artifact.snapshot.datasets;
data.validation_status = [
  { status: "Valid and MAPE ≤ 40%", models: 65, definition: "Holdout MAPE is calculable and no greater than 40%" },
  { status: "High error", models: 9, definition: "Holdout MAPE exceeds 40%" },
  { status: "Unscorable", models: 6, definition: "Holdout MAPE is missing" }
];

data.architecture_options = [
  { rank: 1, option: "Hybrid architecture (D)", role: "Portfolio-level architecture", benefit: "Combines current compatibility, a regularized default path, and targeted Bayesian upgrades", risk: "Requires clear upgrade and fallback governance", decision: "Recommended target architecture" },
  { rank: 2, option: "Constrained regularized MMM (B)", role: "Future default champion core", benefit: "Improves out-of-sample stability under collinearity while preserving signs and bounds", risk: "Solver, penalty, and constraint design require nested validation", decision: "Phase 3 challenger" },
  { rank: 3, option: "Improved current constrained linear model (A)", role: "Transition layer and fallback", benefit: "Low-risk improvements to validation, temporal baseline, and diagnostics", risk: "Keeping Stepwise as the primary selector limits stability gains", decision: "Retain through Phases 0–2" },
  { rank: 4, option: "Bayesian/hierarchical MMM (C)", role: "Upgrade layer for high-value or high-risk slices", benefit: "Represents uncertainty, partial pooling, and experiment calibration", risk: "Weak identification at roughly 104 weeks plus high runtime and prior-governance cost", decision: "Phase 4 pilot only" }
];

data.roadmap = [
  { phase_order: 0, phase: "Phase 0", scope: "Tests, baseline freeze, data fingerprint, and deterministic splitter", deliverable: "Reproducible 80-model baseline and test suite", gate: "All baseline tests are stable and existing outputs remain untouched", approval: "Approve now" },
  { phase_order: 1, phase: "Phase 1", scope: "Eligibility, rolling validation, MASE/WAPE/MAE/bias, and grading", deliverable: "Baseline validation report and threshold sensitivity", gate: "Current coefficients remain unchanged and every model has an explicit scoreability status", approval: "Approve now" },
  { phase_order: 2, phase: "Phase 2", scope: "Trend/Fourier/event baseline and residual diagnostics", deliverable: "Temporal-baseline challenger results", gate: "Outer error does not regress and residual structure improves", approval: "Wait for Phase 1 results" },
  { phase_order: 3, phase: "Phase 3", scope: "Constrained Ridge/Elastic Net and composite dead-variable pruning", deliverable: "Champion–challenger portfolio comparison", gate: "Generalization, tail performance, stability, and business constraints all improve", approval: "Separate approval" },
  { phase_order: 4, phase: "Phase 4", scope: "Targeted Bayesian/hierarchical pilot and calibration interface", deliverable: "Posterior and cost report for a small pilot set", gate: "Clearly outperforms the frequentist champion at acceptable maintenance cost", approval: "Separate approval" },
  { phase_order: 5, phase: "Phase 5", scope: "Versioning, drift, review workflow, rollback, and runtime monitoring", deliverable: "Operable model lifecycle", gate: "Prior algorithms and interfaces are stable", approval: "Outside current scope" }
];

data.quality_framework = [
  { grade_order: 1, grade: "Green", meaning: "Sufficient external validation and stability", candidate_rule: "Full eligible; multiple valid outer folds; candidate MASE/WAPE/bias, residual, and stability gates all pass", allowed_use: "Explanation, scenario analysis, and directional planning", action: "Automatically pass; does not imply causal calibration" },
  { grade_order: 2, grade: "Yellow", meaning: "Directionally useful but evidence is limited", candidate_rule: "At least one valid fold; moderate error, residual, or stability concerns; Limited models are capped at Yellow", allowed_use: "Explanation after manual review", action: "No automated budget optimization" },
  { grade_order: 3, grade: "Red", meaning: "Fit is possible but statistical evidence is inadequate", candidate_rule: "Materially worse than benchmark, large bias, serious residual failure, or unstable key contributions", allowed_use: "Diagnostics only", action: "Reject for business release" },
  { grade_order: 4, grade: "Unscorable", meaning: "The model is not eligible to be graded", candidate_rule: "No valid holdout, all-zero or near-zero target, insufficient history/nonzero weeks, or no target variation", allowed_use: "Descriptive or pooled-only route", action: "Exclude from the comparable model pool" }
];

data.file_plan = [
  { order: 1, surface: "tests/ (new)", planned_change: "Unit, integration, baseline, and leakage-prevention tests", compatibility: "Do not write to existing outputs; pin input fingerprints", phase: "0" },
  { order: 2, surface: "validation module (new)", planned_change: "Rolling splitter, robust metrics, and quality rules", compatibility: "Run beside the current path first; do not alter run_slice results", phase: "0–1" },
  { order: 3, surface: "code/capstone_pipeline.py", planned_change: "Later connect validator and estimator strategies through a compatible entry point", compatibility: "Preserve run_slice parameters and current return fields", phase: "1–3" },
  { order: 4, surface: "estimator module (new)", planned_change: "Wrap the baseline and constrained-regularization challenger", compatibility: "Preserve contribution calculations and variable names", phase: "3" },
  { order: 5, surface: "code/run_all.py", planned_change: "Append grade and validation fields and emit experiment comparisons", compatibility: "Keep existing columns; append new fields", phase: "1–3" },
  { order: 6, surface: "code/dashboard.py", planned_change: "Later display grade, folds, and stability; no first-round change", compatibility: "Resolve the authoritative-summary source before consuming new fields", phase: "Separate follow-up" },
  { order: 7, surface: "requirements and lock files", planned_change: "Pin the environment; require separate approval for new solver dependencies", compatibility: "Preserve a fallback without heavy dependencies", phase: "0 and 3" }
];

data.risk_register = [
  { risk_order: 1, risk: "Time leakage", impact: "Creates false outer-fold improvement", control: "Learn every transformation, scaling, selection, and pruning decision inside training folds only", stop_condition: "Any outer data used for tuning invalidates the experiment" },
  { risk_order: 2, risk: "Attribution drift", impact: "Temporal baseline or regularization reallocates media Due-to", control: "Compare reconstruction, top-driver stability, and business signs together", stop_condition: "Do not upgrade if major contribution shifts cannot be explained" },
  { risk_order: 3, risk: "Oversized candidate space", impact: "Overfits adstock and saturation on roughly 104 weeks", control: "Use a compact predefined grid and a one-standard-error simplicity rule", stop_condition: "In-sample R² rises without outer-fold improvement" },
  { risk_order: 4, risk: "Arbitrary quality thresholds", impact: "Incorrectly releases or rejects many models", control: "Run Phase 1 threshold sensitivity and require owner approval", stop_condition: "Classifications change materially under small threshold changes" },
  { risk_order: 5, risk: "Dependency and runtime expansion", impact: "Windows environment becomes irreproducible or the 80-model batch becomes too slow", control: "Approve dependencies by phase, pin versions, and preserve fallback", stop_condition: "Runtime cost exceeds budget without material quality gain" },
  { risk_order: 6, risk: "Prediction gains described as causal gains", impact: "Budget and ROI conclusions are overstated", control: "Use model-implied contribution consistently and add a separate calibration label for experiments", stop_condition: "Causal claims appear without identification evidence" }
];

data.approval_decisions = [
  { decision_order: 1, decision: "Current authorization scope", recommended: "Phases 0–1 only", why: "Build evaluation capability before choosing an estimator replacement" },
  { decision_order: 2, decision: "Protection of current outputs", recommended: "Write experiment results to a separate directory", why: "Avoid overwriting the current 80-model and demo results" },
  { decision_order: 3, decision: "Eligibility thresholds", recommended: "Treat them as candidates; do not hard-code yet", why: "The 91-week and 40-nonzero-week thresholds have not been sensitivity-tested across the portfolio" },
  { decision_order: 4, decision: "Success criteria", recommended: "Use outer error, tail performance, stability, and pass rate together", why: "Prevent optimization for higher R² or a single median" },
  { decision_order: 5, decision: "New solver dependency", recommended: "Approve CVXPY only after a Phase 3 prototype wins", why: "Avoid expanding the environment and maintenance scope prematurely" },
  { decision_order: 6, decision: "Bayesian scope", recommended: "Small Phase 4 pilot", why: "Current data length and engineering objectives do not justify a first-round full rewrite" }
];

artifact.manifest.sources.forEach((source) => {
  if (source.id === "deep_research") source.label = "User-provided deep-research-report.md";
});
artifact.package_info.originUrl = "artifact://mmm-modeling-engine-improvement-proposal-en";

fs.writeFileSync(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
console.log(outputPath);

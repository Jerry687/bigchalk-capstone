# Big Chalk — NU Summer 2026 Capstone: Project Brief

*Compiled from the kickoff email thread, the intro deck, and the data file. Last updated: 2026-06-23.*

## The one-liner
Build an **automated regression engine + UI** that models a CPG client's **Volume Sales** for every **Brand × Retailer-Channel** combination, then wrap it in a dashboard for non-technical users to run, tweak, and finalize models.

Deck title: *"When one model just isn't enough: Application of automated modeling techniques to high granularity datasets with UI Integration."*

## Who
- **Sponsor:** Big Chalk Analytics (Evanston, IL). End client is an unnamed national CPG manufacturer/distributor with many niche/flavored products across many channels.
- **Big Chalk contacts:** Alex Hathcock (Senior Data Scientist — main contact), Arko Mandal (Lead Data Scientist), Sean Ogar (CEO/Co-Founder), John Parkins (Head of HR / NDAs).
- **Faculty:** Abid Ali (Northwestern).
- **Team:** Feifan Liu, Boqi Niu, Jiahao Li (Edison).

## Timeline
- Runs **Mon Jun 22 → end of August 2026**.
- **Final presentations: Tue Aug 25, 10 AM** (in person encouraged).
- First check-in with Alex: ~**Wed Jul 1** ("Wednesdayish next week").

Suggested phasing from the deck:
1. Understand the data — 1 week
2. Understand & automate one model from scratch — 1 week
3. Expand to all Product×Channel models (scalability) — 1 week
4. Design the UI/dashboard — 3–4 weeks
5. "Brag about it" — build the final presentation — 1 week

## The Ask (functional requirements)
An automated regression solution that:
- Models **Product × Retailer Channel** data with predictors spanning **competition, media, trade, and price**.
- Builds each model from the ground up via **automated techniques**: stepwise/elimination, regularization/shrinkage, random forest / gradient-boosting importance, or custom.
- Allows **manual override at the end**: select variable inclusion/exclusion and set **coefficient bounds** (positive / negative / unconstrained / a custom range excluding 0).
- Outputs **coefficients, standard fit metrics, variable contributions ("due-tos"), VIF, t-stats, fit charts**.
- Bundles the whole flow into a **dashboard (Plotly or similar)**: data upload → model design → output, with drill-down to individual Product×Channel models, manual rerun, and easy viewing.

**Modeling note from Alex:** the final model must support coefficient constraints, which makes pure Ridge awkward for the *final* model — anything goes while exploring. Volume Sales is modeled across time by Brand×Channel; week/date variables mainly capture seasonality/trend and separate observations.

## The Data — `Anonymized Data for Project.xlsx`
- 11 tabs: **"General Data Dictionary"** + **Brand 1 … Brand 10** (one sheet per brand).
- **~3 years of weekly data**, week-ending dates spanning **Jan 2023 → Dec 2025** (156 weeks).
- Each brand sheet ≈ **1,248 rows** (≈9 channels × 156 weeks), **113 columns**. (Brand 9 slightly fewer; Brand 6 has some `#N/A` channels.)
- Grain: one row per **Channel (Geography) × Brand (Product) × Week**.
- Up to **9 channels** ("Channel 1"–"Channel 9"); brands cover varying subsets.

### Column families (per data dictionary)
- **Keys:** Geography (= Channel, modeling dimension), Product (= Brand), Time / Week / Week_Num (1–52).
- **Target:** `Volume Sales` (dependent variable).
- **Trade:** Volume Sales by merch condition (No Merch, Any Merch, Price Reductions, Feature, Display, Special Pack, Feature & Display) + **Weighted Weeks** versions (reach/frequency of trade executions).
- **Distribution:** Avg Weekly Items per Store Selling (depth), ACV Weighted Distribution (breadth), Total Points of Distribution (and TPD × trade conditions).
- **Price:** Price per Volume (overall + by trade condition).
- **Other POS:** Dollar Sales (overall + by trade condition).
- **Category/competitive price:** Category P Price per Volume, Total Category Price per Volume.
- **Macro-economic (~18):** Unemployment_Rate, Median_CPI, CFNAI, Gas_Price, UMCSENT, T5YIE, PSAVERT, PCEC96, SNAP_* (5), Fedfunds, Unempclaims, Retail_Sales.
- **Trend & Seasonality:** Trend, Seasonality_Index.
- **Media spend (12 channels):** Meta, TikTok, Google Ads, Walmart, Amazon, Instacart, Doordash, Absco, Ahold, Kroger, Instacart Display, Instacart CTV. **Should be decayed / adstocked** (per dictionary + Alex).
- **Competitive data:** Competitor1–6 distribution (Items/Store, ACV) and trade (Weighted Weeks) variables.

## Files in this folder
- `Anonymized Data for Project.xlsx` — the dataset (data dictionary + 10 brand tabs).
- `Big Chalk Summer 2026 NU Capstone Project.pptx` — the kickoff deck.
- `Project_Brief.md` — this summary.

# Feasibility Framework: Data Quality & Predictive Viability (Pre-Modeling)

This framework is meant to be used before model building. The goal is to reduce wasted effort and avoid deploying models that look strong in a notebook but fail under real-world constraints.

## Step 1 — Clarify the prediction and decision
- What decision will the model support?
- What is the target, and when is it observed?
- What is the prediction horizon (how far ahead)?
- What does “success” mean (cost/benefit, risk, operational use)?

## Step 2 — Data profiling and basic EDA
- Basic stats: null %, unique %, min/max, distribution shape
- Outliers and impossible values (domain sanity checks)
- String fields: inconsistent categories, casing, formatting
- Time fields: ordering issues, missing timestamps, clock skew

## Step 3 — Data integrity
- Primary key uniqueness and duplicate handling
- Join integrity: match rates, many-to-many risks
- Rule-based validations (ranges, allowed categories, referential integrity)

## Step 4 — Leakage checks (high risk)
Common leakage patterns:
- Features created after the outcome occurs
- Aggregations that include future events
- Fields that encode the label indirectly (status flags, resolution codes)
- Train/test splits that ignore time ordering for time-dependent outcomes

## Step 5 — Predictive signal and interpretability
- Does the target show separability at all?
- Are signals consistent across segments and time?
- Are features interpretable enough to trust in production decisions?

## Step 6 — Stability / drift indicators
- Compare feature distributions across time windows
- Compare across key segments (region/store/device/customer cohorts)
- Flag features likely to drift due to policy/process changes or seasonality

## Step 7 — Communicate and decide
Output a short, decision-ready summary:
- Is modeling feasible right now? (Yes/No/Conditional)
- Biggest risks (data gaps, leakage, drift, label quality)
- Recommended next actions (data collection fixes, governance improvements, experiment plan)

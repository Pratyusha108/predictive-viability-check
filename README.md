# Assessing Data Quality & Predictive Viability Before Model Development

In many real-world analytics projects, models fail not because of algorithm choice, but because the data was never suitable for prediction in the first place.
This repo documents a practical, repeatable framework for evaluating whether a dataset is ready for predictive modeling before investing time in ML.

## Why this exists
Most teams jump straight to modeling. I wanted a structured way to answer a more fundamental question first:

**Should we model this at all?**

This project focuses on early-stage feasibility: data quality, stability, leakage risk, and business interpretability—so modeling decisions are grounded, explainable, and reliable.

## Objectives
- Evaluate **data availability, quality, and relevance** before model development
- Identify **signal vs. noise**, leakage risks, and bias early
- Assess feature stability (time / segment) to reduce production failure risk
- Translate findings into **stakeholder-ready** decision notes

## Framework (high level)
### 1) EDA & data profiling
- Missingness, sparsity, outliers, inconsistent formats
- Distribution sanity checks and basic statistical profiling

### 2) Data quality & integrity
- Duplicate keys, impossible values, timestamp integrity
- Join/merge reliability and rule-based validation

### 3) Predictive viability
- Weak signal detection (target separation)
- Feature usefulness vs. business interpretability

### 4) Leakage risk checks
- Features that indirectly reveal the target
- Post-outcome features, time leakage, and “future info” contamination

### 5) Stability & drift indicators
- Segment stability (location/store/device/customer cohorts)
- Time-based stability and drift-aware feature review

### 6) Governance & trust
- Document assumptions and limitations
- Make outcomes explainable and auditable for decision-makers

## Key takeaways
- High model accuracy can be meaningless without validating data quality and stability first
- Many “predictive” signals come from leakage or temporal artifacts
- Feasibility analysis prevents wasted modeling effort and improves stakeholder trust

## Repo contents
- `docs/feasibility-framework.md` — full framework + reasoning
- `docs/checklist.md` — quick checklist to reuse on any dataset
- `notebooks/` — example profiling, leakage checks, stability checks (optional)
- `src/` — reusable helper scripts (optional)

## Skills & concepts used
EDA, statistical profiling, missingness analysis, leakage detection, stability checks, stakeholder communication, documentation, and governance-aware analytics.



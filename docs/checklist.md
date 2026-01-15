# Pre-Modeling Checklist 

## Data readiness
- [ ] Target definition is clear (what, when, horizon)
- [ ] Label quality checked (missing, noisy, inconsistent)
- [ ] Missingness analyzed and explained
- [ ] Outliers and impossible values reviewed
- [ ] Keys are unique (or duplicates handled)
- [ ] Joins validated (match rate and cardinality)

## Leakage & validity
- [ ] Time-aware split planned (if time-dependent)
- [ ] Post-outcome features removed
- [ ] Aggregations checked for “future info”
- [ ] Proxy labels and indirect target encodings reviewed

## Signal & stability
- [ ] Quick signal sanity checks performed
- [ ] Segment stability checked (major cohorts)
- [ ] Time stability checked (recent vs older)
- [ ] Drift risks documented (process/policy/seasonality)

## Governance & delivery
- [ ] Assumptions documented
- [ ] Limitations documented
- [ ] Decision recommendation written (go/no-go + next steps)

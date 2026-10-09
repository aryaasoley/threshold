# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

### Added
- First results on the real data (Feedzai BAF `Base.csv`, months 6–7 as the test): the model catches
  54% of fraud at a 5% false-positive rate (ROC AUC 0.891); Threshold adds about £2.1k profit per
  1,000 applications over the best single cut-off; summary tables in `artifacts/`, figures, results
  page and model card in `reports/`.

### Changed
- Docs no longer assume data sources overlap: on the real data the three vendors complement each
  other, so together they are worth more than their separate values added up.
- Chart labels: sensitivity bars are labelled at each end; capacity tile reads "at capacity" when
  demand is within 2% of the team's slots.

## [0.1.0]

### Added
- SQL layer in DuckDB: cleaning (−1 sentinels → NULL), time-based split, monthly KPIs by channel,
  device and age group, and policy segment KPIs.
- Fraud model: logistic-regression baseline and gradient boosting (LightGBM, or scikit-learn's
  histogram GBM where LightGBM can't load), early stopping on month 4, Platt calibration on month 5.
- Decision engine: four actions priced as straight lines in p, exact band edges, capacity-aware
  review allocation with a shadow price, compared with approve-all, a 5%-FPR cut-off and the best
  single cut-off on months 6–7.
- Value of each extra analyst and a recommended team size.
- Value of each data source by retraining without it, with paired-bootstrap ranges.
- Fairness comparison by age group on every policy, with a no-age model.
- Monitoring: monthly replay, actual vs expected fraud by band, score and feature PSI.
- Bootstrap ranges and a sensitivity chart for every assumption.
- Streamlit app: Policy, Operations, Vendors, Monitoring and fairness.
- Tests: known answers, Hypothesis properties, data and leakage checks, end-to-end pipeline on
  generated data; CI on every push.
- Docs: methodology, decision log, experiment plan, assumptions table, model card, data notes.

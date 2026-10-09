# Model card: Threshold fraud score

## What it is
A gradient-boosted tree model (sklearn-hgb, 275 trees, chosen by early stopping on month 4)
estimating the probability that a bank-account application is fraudulent, followed by Platt
calibration fitted on month 5 (a = 1.009, b = 0.411).
It feeds a decision policy; it is not a decision on its own.

## Data
Feedzai Bank Account Fraud (NeurIPS 2022), `Base.csv`: 1M synthetic-but-realistic applications
over 8 months, 30 features. See `DATA.md`. Learn on months 0–4, calibrate on month 5, test on
months 6–7. Missing values (−1 sentinels) are converted to NULL in SQL and left for the trees to route.

## Performance on the test months
| | Fraud caught at 5% FPR | ROC AUC | Average precision | Brier | Mean predicted vs observed |
|---|---|---|---|---|---|
| Baseline (logistic) | 50.9% | 0.881 | 0.163 | 0.0127 | 1.14% vs 1.40% |
| GBM, raw | 54.3% | 0.891 | 0.192 | 0.0125 | 0.99% vs 1.40% |
| GBM, calibrated | 54.3% | 0.891 | 0.192 | 0.0125 | 1.38% vs 1.40% |

"Fraud caught at 5% FPR" is the dataset paper's headline metric. Calibration does not change
the ranking (so ROC AUC is unchanged); it makes the probabilities usable as £ inputs.

## Fairness
At the paper's 5%-FPR cut-off, genuine applicants aged 50+ are flagged
13.7% of the time vs 4.3% for under-50s
(ratio 3.16×). Retraining without age gives 2.30×: other features carry
much of the same information. Age-specific cut-offs are not used (Equality Act 2010); the gap is
reported on every policy for Compliance to judge.

## Limits
- The data is partly synthetic and US-flavoured; the method transfers, the numbers do not.
- The month-to-month fraud rate drifts; calibration is fitted on one month and needs refitting.
- Reviewer and verification outcomes (r, f, a_g, a_f) are assumptions, not measurements.
- Confirmed fraud labels arrive late in practice; PSI is the early warning, not outcome metrics.

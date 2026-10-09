# Assumptions and sources

Generated from `config/assumptions.yaml` by `make report`. Edit the YAML, not this file.

| Symbol | Assumption | Default | Range tested | Unit | Source | Checked |
|---|---|---|---|---|---|---|
| V | Value of a good customer (£) | 50 | 25 – 100 | £ per genuine customer onboarded | Revolut Group Holdings Annual Report 2025: profit before tax £1.7bn across 68.3m retail customers ≈ £25 per customer per year. Times ~2 years of expected active tenure ≈ £50. A public anchor for a digital bank, not this bank's figure. | 2026-10-08 |
| L | Loss per fraud (£) | 1000 | 500 – 2000 | £ per fraudulent account opened | PSR APP-scam reimbursement (in force from 7 Oct 2024): the receiving PSP pays 50% of each reimbursed claim, capped at £85,000 per claim. Assumption: ~£1,700 of scam money lands in a mule account before it is caught → £850 refund share, plus ~£150 investigation and closure cost ≈ £1,000. The rules are under review in 2026–27. | 2026-10-08 |
| s | Verification cost (£ per check) | 1.5 | 0.5 – 3 | £ per verification | Assumption: typical list-price range for a document-and-selfie identity check. | – |
| a_g | Genuine drop-off at verification | 0.15 | 0.05 – 0.3 | share of genuine applicants who abandon when asked to verify | Assumption: one extra onboarding step. Measure it with the A/B test in docs/experiment_plan.md. | – |
| a_f | Fraudster drop-off at verification | 0.7 | 0.4 – 0.9 | share of fraudsters who abandon or fail when asked to verify | Assumption: stolen or synthetic identities struggle with a liveness check. | – |
| r | Share of fraud reviewers catch | 0.8 | 0.6 – 0.95 | share of fraudsters a reviewer stops | Assumption. No public data; measure from QA sampling of review outcomes. | – |
| f | Share of genuine reviewers reject | 0.05 | 0.01 – 0.15 | share of genuine applicants a reviewer wrongly rejects (or who leave while waiting) | Assumption. | – |
| analyst_hourly_cost | Analyst cost (£ per hour, loaded) | 28 | 20 – 40 | £ per hour, fully loaded | Assumption: a UK financial-crime analyst salary of roughly £32–38k plus ~30% on-costs, over ~1,650 working hours a year. | – |
| minutes_per_case | Minutes per review | 12 | 6 – 25 | analyst minutes per review | Assumption. | – |
| productive_share | Share of paid time on cases | 0.8 | 0.65 – 0.9 | share of paid hours spent on cases | Assumption: breaks, training, team meetings. | – |
| hours_per_day | Paid hours per day | 7.5 |  | paid hours per analyst per day | Standard UK office day. | – |
| tooling_cost_per_case | Tooling cost (£ per review) | 1 | 0 – 3 | £ per review (data pulls, case-management licence) | Assumption. | – |
| team_size | Review analysts | 6 |  | analysts on the onboarding review queue today | Assumption: the starting point the operations page re-plans from. | – |
| working_days_per_month | Working days per month | 21 |  | days | Calendar average. | – |

# Decision log

Each entry: the choice, what else was on the table, and why. Newest decisions are added at the end.

### 1. Feedzai's Bank Account Fraud dataset, not made-up onboarding data
**Alternatives:** a simulated onboarding funnel; app-store reviews; another public fraud set
(e.g. card transactions). **Why:** it is account *opening*, at a realistic scale (1M rows, 8 months),
with the real difficulties: rare fraud, drift between months and an age-group disparity. A simulated
funnel would only "discover" what was put into it.

### 2. DuckDB for the SQL layer
**Alternatives:** PostgreSQL, SQLite, pandas only, dbt. **Why:** full analytical SQL over a 1M-row
CSV with no server to install, so anyone can run it. dbt earns its keep with dozens of models,
tests and environments; four SQL files don't need it.

### 3. Split by time: learn 0–4, calibrate 5, test 6–7
**Alternatives:** random split; cross-validation. **Why:** a random split leaks later months' patterns
into training and overstates performance on a drifting problem. One fixed split, decided before any
results were seen, also makes "no test-month tuning" easy to check (`tests/test_data.py`).

### 4. Sentinels become NULL in SQL, not in Python
**Alternatives:** fix them in pandas; leave them for the trees. **Why:** cleaning belongs in one place
upstream of everything, including the KPI tables. A pipeline check fails if a −1 survives.

### 5. Calibrate instead of rebalancing classes
**Alternatives:** SMOTE, class weights, undersampling. **Why:** cut-offs are set in pounds, so p must be
a true probability. Rebalancing distorts p; calibration on a later month also corrects for drift
between training and today.

### 6. Platt scaling, not isotonic regression
**Alternatives:** isotonic, beta calibration. **Why:** month 5 holds roughly a thousand frauds; a
two-parameter fit can't overfit it and keeps the ranking. Isotonic would be a step function with
few points in the high-risk tail, which is exactly where the cut-offs sit.

### 7. Early stopping on month 4, then refit on months 0–4
**Alternatives:** stop on month 5; fixed number of trees; train on 0–3 only. **Why:** month 5 is
reserved for calibration and cut-offs, so it can't also choose the model. The vendor and no-age
variants reuse the full model's number of trees so their comparison isn't muddied by different
stopping points.

### 8. LightGBM, with scikit-learn's histogram GBM as a fallback
**Alternatives:** XGBoost; only scikit-learn. **Why:** LightGBM is the standard for tabular data and
runs in CI. On macOS it needs the OpenMP runtime (`brew install libomp`); without it the pipeline
uses `HistGradientBoostingClassifier`, the same algorithm family, and records which ran in
`artifacts/model_metrics.json`.

### 9. Four actions priced as straight lines in p
**Alternatives:** two actions (approve/decline at one cut-off); a score-band rulebook. **Why:** the
middle options are where the money is, and linearity makes the policy explainable on one chart:
bands, with edges where lines cross. Reviewer and verification outcomes enter as expected values
because the data has no outcomes for them; they are labelled assumptions.

### 10. Capacity by sorting on review gain, not an LP solver
**Alternatives:** scipy `linprog`; a fixed review quota by score rank. **Why:** every review uses one
slot, so greedy-by-gain is provably optimal and gives the shadow price for free. Ranking by score
instead of gain would review the riskiest cases, which are often better simply declined.

### 11. Analyst time is part of c; analyst value adds it back
**Alternatives:** c = tooling only, with salaries as a separate fixed cost everywhere. **Why:** c should
reflect what a review really costs when choosing who to review (the brief's definition). When valuing
a whole analyst, salary is sunk per day, so the step in profit is measured with analyst time added
back and compared with a day's cost. Both views give the same hiring answer at the margin.

### 12. Cut-offs fixed on month 5, capacity enforced again on the test months
**Alternatives:** re-optimise on each test month; let the queue overflow silently. **Why:** fixing the
cut-offs is the honest out-of-sample test. In real operations a queue that overflows is worked in
priority order, so the test months do the same and the app reports reviews wanted against capacity.

### 13. Ship a 1,000-bucket histogram to the app, not scored rows
**Alternatives:** score in the app; ship scored rows. **Why:** every number is a sum over applicants
that depends only on p, so the histogram loses nothing that matters, keeps the app instant, and
keeps row-level data off the hosted app (and out of the repo).

### 14. Value data sources by retraining without them
**Alternatives:** SHAP or permutation importance. **Why:** importance measures say how much the model
leans on a feature, not how much money the decision loses without it. Retrain-and-reprice answers
the vendor question directly, in £ per application. The paired bootstrap gives a range.

### 15. Report the age gap; don't adjust cut-offs by age
**Alternatives:** group-specific cut-offs to equalise false-positive rates; drop age and call it fixed.
**Why:** age is a protected characteristic under the Equality Act 2010, and treating groups
differently by design creates its own legal problem. The right output is a clear, flagged
comparison for Compliance, plus evidence on whether removing age helps (it often doesn't).

### 16. PSI against the training months for features, month 5 for the score
**Alternatives:** one reference for everything. **Why:** feature drift matters relative to what the
model learned from; score drift matters relative to where the cut-offs were set.

### 17. Size the team on month 5 at expected outcomes
**Alternatives:** realised profit on the test months for each team size. **Why:** the realised step
from one analyst rests on a handful of fraud cases, so the "worth hiring?" answer flipped back and
forth between neighbouring team sizes in testing. Planning on month 5 at calibrated expected values
is smooth and concave, uses the same month as every other choice, and the test months' realised
profit is still reported for each team size.

### 18. Left out on purpose
Docker, Kubernetes, Kafka, Spark, Airflow, FastAPI, React, deep learning, any AI features. None
would improve a single decision here, and the project is a tool for *setting* policy, not the live
system that runs it.

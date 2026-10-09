# Methodology

Everything here can be done on a whiteboard in a couple of minutes. The code follows the
same steps, in the same order, and the module for each step is named in brackets.

## 1. Test on the future, not a random shuffle  `[sql/02_features.sql, model.py]`

| Months | Role |
|---|---|
| 0–4 | Learn the model. Month 4 doubles as the early-stopping check, then the model is refit on 0–4. |
| 5 | Calibrate the probabilities and fit every policy's cut-offs. |
| 6–7 | Test. Nothing is tuned here; every number in the README comes from these two months. |

A random split lets the model see later months' patterns during training, so it looks better
than it will be in production. The data also drifts: the fraud rate moves month to month.
Testing on later months measures that honestly.

**The −1 trap.** Several columns use −1 for "missing" (and `intended_balcon_amount` uses any
negative number). Treated as a number, −1 months at an address is a silent bug. `sql/01_clean.sql`
turns every sentinel into NULL, `threshold.data.check_data` fails the pipeline if one survives,
and `tests/test_data.py` checks both.

## 2. Calibration: make the probabilities honest  `[model.py]`

Cut-offs are set in pounds, so p must mean what it says: of applicants scored 4%, about 4%
should turn out to be fraud. The gradient-boosted model's raw scores are rescaled with
**Platt scaling** fitted on month 5:

    p = sigmoid(a · logit(raw score) + b)

Two parameters can't overfit a month with ~1,000 frauds, and the ranking is unchanged (ROC AUC
is identical before and after). The check is a reliability chart and the Brier score on the test
months (`reports/figures/calibration.png`).

**Why not oversample fraud (SMOTE) or re-weight classes?** Both change the base rate the model
learns, so its probabilities stop meaning what they say. The decision layer needs honest p more
than it needs a slightly higher recall at one cut-off.

## 3. The core idea: each action's value is a straight line  `[economics.py]`

For an applicant with fraud probability p, every action's expected value has the form

    EV(p) = (1 − p)·G − p·B − k

| Action | G: value if genuine | B: cost if fraud | k: fixed cost |
|---|---|---|---|
| Approve | V | L | 0 |
| Verify | (1 − a_g)·V | (1 − a_f)·L | s |
| Review | (1 − f)·V | (1 − r)·L | c |
| Decline | 0 | 0 | 0 |

V = value of a good customer, L = cost of a fraudster, c and s = cost of a review and of a
verification check, r = share of fraudsters a reviewer stops, f = share of genuine applicants a
reviewer wrongly rejects, a_g and a_f = share of genuine applicants and of fraudsters who drop
out when asked to verify. Every value, its range and its source is in `config/assumptions.yaml`
(rendered in [assumptions.md](assumptions.md)).

**Pick the highest line at each p.** The lines' slopes are −(G + B), and the upper envelope of
straight lines is convex, so as p rises the winning line can only get flatter. Each action wins
over **one unbroken band**, and the bands come in order of steepness: usually approve, verify,
review, decline. `economics.envelope` walks the envelope exactly; `tests/test_policy_properties.py`
checks the band order on thousands of random assumption sets.

**Where the bands meet** (set two lines equal and solve for p):

| Boundary | p |
|---|---|
| approve = decline | V / (V + L) |
| approve = verify | (a_g·V + s) / (a_g·V + a_f·L) |
| verify = review | ((f − a_g)·V + c − s) / ((f − a_g)·V + (r − a_f)·L) |
| review = decline | ((1 − f)·V − c) / ((1 − f)·V + (1 − r)·L) |

**Worked example (the defaults).** V = £50, L = £1,000, c = £8, s = £1.50, r = 0.8, f = 0.05,
a_g = 0.15, a_f = 0.7:

- approve below (7.5 + 1.5) / (7.5 + 700) = **1.27%**
- verify from 1.27% to (−5 + 6.5) / (−5 + 100) = **1.58%**
- review from 1.58% to (47.5 − 8) / (47.5 + 200) = **15.96%**
- decline above 15.96%

With only approve and decline you would decline everyone above V / (V + L) = 50 / 1,050 = **4.76%**.
The middle options let you keep many of the genuine applicants between 1.6% and 16% instead of
turning them all away.

**Realised value on labelled test data.** For a test applicant whose label y is known, the value
of the action taken is G − k if genuine and −B − k if fraud. Reviewer and verification outcomes
are not in the data, so they enter as expectations (e.g. a reviewed fraudster costs (1 − r)·L + c).

## 4. Capacity, and the value of one more analyst  `[policy.py]`

If the review band holds more cases than the team can handle, which cases get a review? Each
review takes exactly one slot, so this is a knapsack problem with equal weights, and the greedy
answer is optimal: **sort cases by the gain a review adds over the next-best action, and fill
the slots from the top.**

    gain(p) = EV_review(p) − max(EV_approve, EV_verify, EV_decline)(p)

The gain is a straight line minus a convex function, so it is concave. It peaks in the middle of
the review band, and the cases above any gain level form one interval. The cases that get
reviewed are still one unbroken band, just narrower; the rest fall back to verify (low side)
or decline (high side). A property test checks this on random inputs.

**The shadow price.** The gain of the best case left without a slot, λ, is what one more slot
would add. In the code, c includes the analyst's time (hourly cost × minutes per case, grossed
up for non-case time), so λ is net of that time. One more slot is worth λ + (analyst time per
case) before paying for it.

**Hiring.** Salaries are paid whether or not a slot is used. For each team size n the policy is
re-fitted on month 5 with that many slots and valued at calibrated expected outcomes. The value of
analyst n is the step up in profit with analyst time added back:

    value of analyst n per day = [P(n) + c_time·R(n)] − [P(n−1) + c_time·R(n−1)]   (per working day)

where P is profit and R the number of reviews done. Hire while that beats a day's cost. Because
each analyst works the most valuable cases still unworked, each adds less than the last.

Why expected rather than realised outcomes? One analyst adds a few hundred reviews over a month,
so a realised step in profit rests on a handful of fraud cases and the hire/don't-hire answer
flips on noise. In expectation the curve is concave by construction. The realised profit of each
team size on the test months is shown alongside as a check.

**A neat consequence.** c already charges each review the analyst's time, so one more slot is
worth (λ + c_time) and a day of slots is worth about (λ + c_time) × cases per day, against a cost
of c_time × cases per day. The next analyst pays for themselves exactly when λ > 0, i.e. while
the review band still holds more worthwhile cases than the team can work. The first analyst who
would sit partly idle is the one not worth hiring.

**Out of sample.** The cut-offs (and λ) are fixed on month 5. If more test-month cases qualify
than the team can handle, analysts work the most valuable first and the rest take their next-best
action. The app shows reviews wanted against capacity, so an overflowing queue is visible.

## 5. What each data source is worth  `[vendors.py]`

Features are grouped by who would plausibly supply them: the application form, internal activity
checks, device data, a credit bureau, and email and phone checks. For each group:

1. retrain the model without it (same number of trees), recalibrate on month 5;
2. re-fit the policy on month 5 and score months 6–7;
3. value = profit per application with all data − profit per application without the group.

That value is the break-even price per check: pay more and you lose money. The 95% range comes
from a **paired bootstrap**: resample the same test applicants for both models, so noise common
to both cancels. Sources overlap in what they reveal, so the individual values do not add up;
the pipeline also drops all three external vendors together to show by how much.

## 6. Fairness  `[fairness.py]`

For every policy, the share of **genuine** applicants who are declined, reviewed, asked to verify,
or any of the three, for applicants aged 50+ and under 50, and the ratio between them. The dataset
paper found large gaps on exactly this comparison. Age is a protected characteristic under the
Equality Act 2010, so Threshold does not set different cut-offs by age. It reports the gap, flags
ratios outside 0.8×–1.25×, and, as a check, retrains without age. That rarely closes the gap,
because other features carry the same information.

## 7. Monitoring  `[monitoring.py]`

The test months are replayed one at a time with the policy fixed on month 5:

- approval rate and the share in each band;
- fraud stopped;
- actual vs expected fraud in each band (expected = Σ p of the applicants in it);
- the **population stability index** of the score (against month 5) and of each feature
  (against the training months):

      PSI = Σ over buckets (cur% − ref%) · ln(cur% / ref%)

  Under 0.10 is stable, 0.10–0.25 watch, over 0.25 act.

In production, confirmed fraud arrives weeks late, so outcome metrics lag. PSI needs no labels,
which makes it the early warning.

## 8. Uncertainty  `[uncertainty.py]`

Fraud is ~1% of applications, so the headline numbers rest on a few thousand fraud cases.

- **Bootstrap.** Each repetition resamples the calibration month *and* the test months (Poisson
  weights on the histogram cells, which is equivalent to resampling applicants with replacement),
  re-fits the cut-offs on the resampled month 5 and scores the resampled test months. The 2.5th
  and 97.5th percentiles give the range.
- **Sensitivity (tornado).** Each assumption is swung to the low and high ends of its range in
  `config/assumptions.yaml`, one at a time, and the extra profit over the best single cut-off is
  recorded. The recommendation should hold across every bar: if a bar crosses zero, say so.

## 9. Why a histogram  `[histogram.py]`

Every decision depends on the applicant only through p, and every reported number is a sum over
applicants. So the pipeline packs the scored calibration and test applicants into 1,000
probability buckets (log-spaced, about 1.2% wide in relative terms) with counts of genuine and
fraud applicants per bucket, split by model variant, month and age group. The app re-optimises
from that table in milliseconds, and no row-level data ever reaches the hosted app.

## 10. Benchmarks

- **Model:** share of fraud caught when 5% of genuine applicants are flagged (the paper's metric),
  plus ROC AUC and average precision, against a logistic-regression baseline.
- **Policy:** profit per 1,000 applications against approve-everyone, a single cut-off at 5% FPR
  (as in the paper), and the best single cut-off, all fitted on month 5, with bootstrap ranges.
- **Speed:** stage timings for the full pipeline are written to `artifacts/meta.json`.

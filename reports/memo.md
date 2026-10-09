**To:** Head of Onboarding
**From:** Arya Soley, Strategy & Operations
**Date:** 9 October 2026
**Subject:** Replace the single fraud cut-off with a four-action onboarding policy

## Recommendation

Replace today's single approve/decline cut-off with four actions (approve, ask to verify, send
to review, decline), chosen for each applicant by its expected value in pounds. On two months of
applications the model had never seen, this adds **£2.1k of profit per 1,000 applications**
(95% range £1.9k–£2.4k): about **£2.6m a year** at our current volume of roughly 100k
applications a month.

## Why

- **Fewer good customers turned away.** The best single cut-off turned away 6.4 genuine
  applicants for every fraudster it stopped. The four-action policy turns away 3.5.
- **More fraud stopped and more customers approved.** It stops 67% of fraud (vs 54%) while
  onboarding 96.6% of genuine applicants (vs 95.2%).
- **It holds up.** With every assumption moved to the low and high end of its plausible range,
  the advantage never fell below £1.1k per 1,000 applications.

## What it needs from operations

- **Review team: add four analysts (6 → 10).** Today's team is fully used. Each of analysts
  7–10 returns at least 28% more than their £210 daily cost, about £300 a day net in total.
  Analysts 11–23 still pay for themselves, but by less each time (only 5% above cost by the
  20th), and that rests on an unmeasured 12-minute handling time. Measure it before going further.
- **Data vendors: a ceiling price for each.** Device data is worth £0.63 per application
  (range £0.50–£0.78), email and phone checks £0.40, the credit bureau £0.31. These are
  walk-away prices, not targets. Each figure assumes the other two are kept: together they are
  worth £1.46, more than the £1.34 their separate figures add up to, so don't cut two at once
  on these numbers alone.

## Risks

- **Fairness.** Genuine applicants aged 50+ are given an extra check or declined 2.3× as often
  as under-50s (3.2× under the single cut-off). Removing age from the model narrows this to
  1.8× and costs only £105 per 1,000 applications. I recommend removing it, and taking the
  remaining gap to Compliance rather than setting separate cut-offs by age, which is a
  protected characteristic.
- **Customer friction.** 13% of applicants would be asked to verify. If more genuine applicants
  abandon than the 15% assumed, the gain shrinks.
- **Drift.** The fraud rate rose from 1.2% to 1.5% over the test months, and in the last month
  15% more fraud arrived than the model expected. Recalibrate monthly, and re-fit the cut-offs
  if the gap passes 20%.

## Next step

Test the verify step before a full rollout. An A/B test with about 2,000 applicants per arm can
detect a 3-point difference in genuine drop-off within weeks. The fraud effect needs about
7,750 per arm and has to wait for confirmed fraud cases to come in.

---

*Based on Feedzai's public Bank Account Fraud dataset (1M applications); the bank and its
volumes are illustrative. Method, assumptions and sources: [README](../README.md),
[results](results.md).*

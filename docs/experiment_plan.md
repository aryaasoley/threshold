# Experiment plan: is "verify" really cheaper than declining?

The policy's verify band rests on three numbers that no public data can supply: how many genuine
applicants abandon when asked to verify (a_g), how many fraudsters do (a_f), and what a check costs
(s). The sensitivity chart shows which of them move the answer most. This plan measures them.

## Question
For applicants whose fraud probability lands in the verify band, does asking for a document and
selfie check onboard more genuine customers per £ of fraud loss than the current treatment?

## Design
- **Unit:** the application. Randomise at the moment the score is produced, so assignment can't be
  influenced by anything that happens later.
- **Eligible:** applicants in the verify band, plus a margin either side of it, so the test also tells
  us whether the band edges are in the right place.
- **Arms:** control = today's treatment for that band (e.g. decline or manual review);
  treatment = verify.
- **Primary metric:** share of genuine applicants who complete onboarding (1 − a_g).
- **Secondary:** confirmed fraud among those onboarded (gives a_f); cost per completed check (s).
- **Guardrails:** fraud loss per 1,000 applications; time to open an account; complaint rate.
  Stop early if a guardrail moves the wrong way past a pre-agreed limit.

## Sample size
Two-proportion test, 5% significance (two-sided), 80% power, from `threshold.experiment`:

| What we want to detect | Applicants per arm |
|---|---|
| Genuine drop-off 15% → 12% | 2,036 |
| Genuine drop-off 15% → 10% | 686 |
| Fraud among completers 1.5% → 1.0% | 7,750 |

```python
from threshold.experiment import sample_size_per_arm, weeks_needed

n = sample_size_per_arm(0.15, -0.03)  # 2,036
weeks_needed(n, eligible_per_week=1_000)
```

Drop-off can be read within days. The fraud effect needs nearly four times as many applicants
**and** has to wait for confirmed fraud to arrive (weeks), so plan the test in two reads: an early
read on completion, and a later read on fraud once labels mature.

## Decision rule (agreed before starting)
Plug the measured a_g, a_f and s into `config/assumptions.yaml` and re-run. Keep the verify band if
the measured values keep it ahead of its neighbours across the bootstrap range; otherwise narrow or
remove it.

## Risks
- **Novelty:** completion may dip at first as people meet a new step. Run for at least two full weeks.
- **Interference:** a fraud ring that hits both arms can distort them. Watch device and email
  velocity features by arm.
- **Fairness:** report completion by age group; a check that older applicants abandon more is a
  fairness issue as well as a conversion one.

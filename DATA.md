# Data

## Source

**Bank Account Fraud (BAF) dataset suite, `Base.csv`**, published by Feedzai at NeurIPS 2022
(Datasets and Benchmarks track).

- Kaggle: <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>
- Code and datasheet: <https://github.com/feedzai/bank-account-fraud>
- Paper: Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J. and
  Bizarro, P. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML
  Evaluation.* NeurIPS 2022 Datasets and Benchmarks Track.

1,000,000 bank-account applications over 8 months, 30 features, fraud in roughly 1% of rows. It was
generated from a real, anonymised account-opening dataset with privacy-preserving techniques, so it
keeps the real difficulties (rare fraud, drift between months, group disparities) without describing
real people. Only `Base.csv` is used; the five variants are not.

All credit for the data belongs to Feedzai and the paper's authors.

## Licence

The dataset pages list it under **Creative Commons BY-NC-ND 4.0** (attribution, non-commercial,
no derivatives). **Check the licence on the Kaggle page yourself before publishing anything built
from it.** This repository:

- never commits or redistributes the data (`data/` is ignored by git; `make data` downloads it);
- publishes only aggregate results in `artifacts/`: counts per probability bucket, monthly KPIs and
  summary tables, never a row about an applicant;
- is non-commercial and credits the source here and in the README.

## Getting it

1. Create a free Kaggle account.
2. Authenticate once: `uv run kaggle auth login` (opens a browser), or create an API token under
   Kaggle → Settings → API and save it to `~/.kaggle/access_token`.
3. Accept the dataset's terms on its Kaggle page if prompted.
4. `make data` downloads `Base.csv` (about 200 MB) to `data/raw/`.

## Quirks the pipeline handles

| Column | Raw convention | Cleaned to |
|---|---|---|
| `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `device_distinct_emails_8w` | −1 = missing | NULL |
| `intended_balcon_amount`, `session_length_in_minutes` | negative = missing | NULL |
| `customer_age` | rounded to the decade | kept; `age_group` = 50+ / under 50 |
| `device_fraud_count` | constant 0 in `Base.csv` | kept (carries no information) |

`sql/01_clean.sql` does the cleaning and `threshold.data.check_data` fails the pipeline if a
sentinel survives.

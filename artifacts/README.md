# artifacts/

Small summary tables written by `make train` and read by the app and `make report`: counts of
genuine and fraud applicants per probability bucket, monthly KPIs, and result tables. About 0.5 MB
in total. No row about any applicant is ever written here.

Empty until the pipeline has been run on the real data (`make all`). Results from generated data go
to `artifacts_fake/`, which git ignores.

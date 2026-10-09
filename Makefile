.PHONY: help setup data train report all fake app test lint check clean

PY := uv run python

help:  ## list the targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

setup:  ## install the pinned environment
	uv sync

data:  ## download Base.csv from Kaggle (needs a free Kaggle account; see DATA.md)
	$(PY) -c "from threshold.data import download; download()"

train:  ## SQL layer, model, decisions and analyses -> artifacts/
	$(PY) -m threshold.pipeline

report:  ## figures and Markdown from artifacts/ -> reports/, docs/assumptions.md, README
	$(PY) -m threshold.report

all: data train report  ## everything, from download to results

fake:  ## the whole pipeline on generated data -> artifacts_fake/ (no download; what CI runs)
	$(PY) -m threshold.pipeline --fake --fast
	$(PY) -m threshold.report --artifacts artifacts_fake

app:  ## open the policy simulator
	uv run streamlit run app/Home.py

test:  ## known-answer, random-input, data and pipeline tests
	uv run pytest

lint:  ## style checks
	uv run ruff check .
	uv run ruff format --check .

check: lint test  ## what CI runs before the mini pipeline

clean:  ## remove generated local files (keeps the raw download)
	rm -rf artifacts_fake .pytest_cache .ruff_cache .hypothesis
	rm -f data/*.duckdb

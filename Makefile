.PHONY: help install install-gpu repro repro-spine repro-campaign figures notebook test clean

PY  := .venv/bin/python
# Self-contained: the vendored framework lives under ./src, so both the legacy
# `src.*` imports and the `snptx.*` package resolve from this repo alone.
ENV := PYTHONPATH=$(CURDIR):$(CURDIR)/src

help:
	@echo "snptx-repro-discovery — calibrated sequential decision engine for molecular discovery"
	@echo ""
	@echo "Targets:"
	@echo "  make install        Create .venv and install pinned CPU deps (~5 min, ~1.5 GB)"
	@echo "  make install-gpu    Add CUDA wheels to the same venv (optional; the engine is CPU-only)"
	@echo "  make repro-spine    E1-E5 calibration/SPRT spine + E7 discovery + rule cards (CPU, ~5 min)"
	@echo "  make repro-campaign E8 end-to-end autonomous campaign + timeline/lineage (CPU, ~5 min)"
	@echo "  make repro          repro-spine then repro-campaign"
	@echo "  make figures        Regenerate every figure under preprint/figures/"
	@echo "  make notebook       Execute the guided math-to-figure walkthrough notebook"
	@echo "  make test           Run the fast, no-download determinism regression tests"
	@echo "  make clean          Remove caches and the regenerable DuckDB lineage store"
	@echo ""
	@echo "First run downloads the public TDC ADMET benchmarks (browser User-Agent shim included)."

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

install-gpu:
	$(PY) -m pip install --upgrade torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

repro-spine:
	$(ENV) $(PY) preprint/feasibility_and_figures.py
	$(ENV) $(PY) preprint/e7_discovery_probe.py
	$(ENV) $(PY) preprint/e7_render.py

repro-campaign:
	$(ENV) $(PY) preprint/e8_campaign.py

repro: repro-spine repro-campaign

figures: repro

notebook:
	cd notebooks && ../$(PY) -m jupyter nbconvert --to notebook --execute \
		walkthrough.ipynb --output walkthrough.ipynb \
		--ExecutePreprocessor.timeout=900

test:
	$(ENV) $(PY) -m pytest tests/ -v

clean:
	rm -rf .pytest_cache/ __pycache__/ */__pycache__/ */*/__pycache__/ src/*/__pycache__/
	rm -f preprint/e8_campaign_lineage.duckdb preprint/*_new.json

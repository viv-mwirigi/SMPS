.PHONY: figures scatter report reproducibility env install test

# Environment and reproducibility
reproducibility:
	.venv/bin/python scripts/initialize_reproducibility.py

env:
	conda env create -f environment.yml

install:
	pip install -e .[dev,ml,data,reproducibility]

test:
	pytest tests/ -v --tb=short

# Generate report-ready figures into a single fixed folder (overwrites *.png).
figures:
	.venv/bin/python scripts/plot_latest_figures.py

# Generate scatter-fit evaluation figures (overwrites scatter-fit PNGs).
scatter:
	.venv/bin/python scripts/plot_latest_scatter_fits.py

# Full report set (standard + scatter-fit)
report: figures scatter

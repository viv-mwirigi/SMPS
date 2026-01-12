.PHONY: figures scatter report

# Generate report-ready figures into a single fixed folder (overwrites *.png).
figures:
	.venv/bin/python scripts/plot_latest_figures.py

# Generate scatter-fit evaluation figures (overwrites scatter-fit PNGs).
scatter:
	.venv/bin/python scripts/plot_latest_scatter_fits.py

# Full report set (standard + scatter-fit)
report: figures scatter

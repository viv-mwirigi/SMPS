"""Convenience wrapper to generate scatter-fit figures into a fixed folder.

Usage:
  .venv/bin/python scripts/plot_latest_scatter_fits.py

This reads paired exports from:
- results/ismn_validation/f2pyw/latest/paired_obs_sim.csv
- results/ismn_validation/gdf6e/latest/paired_obs_sim.csv

and writes PNGs into:
- results/validation_figures/latest_figures/ (overwrites scatter-fit PNGs)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate latest scatter-fit figures (short command).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/validation_figures/latest_figures"),
        help="Output folder for figures (scatter-fit PNGs overwritten).",
    )
    parser.add_argument(
        "--f2pyw",
        type=Path,
        default=Path("results/ismn_validation/f2pyw/latest"),
        help="Path to F2PyW latest run folder.",
    )
    parser.add_argument(
        "--gdf6e",
        type=Path,
        default=Path("results/ismn_validation/gdf6e/latest"),
        help="Path to gdf6E latest run folder.",
    )
    parser.add_argument("--n-sites", type=int, default=6,
                        help="Stations per dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "plot_scatter_fit_report.py"

    cmd = [
        sys.executable,
        str(script),
        "--run",
        f"f2pyw={args.f2pyw}",
        "--run",
        f"gdf6e={args.gdf6e}",
        "--out",
        str(args.out),
        "--overwrite",
        "--n-sites",
        str(args.n_sites),
        "--seed",
        str(args.seed),
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

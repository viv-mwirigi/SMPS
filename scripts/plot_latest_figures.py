"""Convenience wrapper to generate the standard validation figures into a fixed folder.

Usage:
  .venv/bin/python scripts/plot_latest_figures.py

This is equivalent to:
  .venv/bin/python scripts/plot_validation_report.py \
    --run f2pyw=results/ismn_validation/f2pyw/latest \
    --run gdf6e=results/ismn_validation/gdf6e/latest \
    --out results/validation_figures/latest_figures \
    --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standard latest figures (short command).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/validation_figures/latest_figures"),
        help="Output folder for figures (overwritten).",
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

    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "plot_validation_report.py"

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
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

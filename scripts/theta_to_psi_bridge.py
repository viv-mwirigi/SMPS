#!/usr/bin/env python
"""Bridge a θ-space model into a ψ-space (kPa) production pipeline.

You said:
- θ-space training performs well
- ψ-space deployment is required
- θ↔ψ conversions introduce errors

This script demonstrates the *safe* way to do it:
1) Keep learning in θ (stable, bounded target)
2) Convert *thresholds* (ψ) to θ once for decisions
3) Convert θ predictions to ψ only for reporting/compatibility, with clipping

Run:
  /home/viv/SMPS/.venv/bin/python scripts/theta_to_psi_bridge.py
"""

from __future__ import annotations
from smps.physics.space import SpaceClips, theta_thresholds_for_psi, theta_to_psi
from smps.physics.van_genuchten import estimate_van_genuchten_params
from smps.core.constants import IRRIGATION_THRESHOLDS

import sys
from pathlib import Path
import numpy as np

# Ensure local imports work when running as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    # Pick a representative soil (replace with your per-plot values in production)
    vg = estimate_van_genuchten_params(
        sand_percent=50, clay_percent=20, organic_matter_percent=2.0)

    print("Van Genuchten params (kPa convention):")
    print(
        f"  theta_r={vg.theta_r:.3f}, theta_s={vg.theta_s:.3f}, alpha={vg.alpha:.4f} 1/kPa, n={vg.n:.3f}")

    # 1) Convert decision thresholds once
    psi_thresholds = {
        "irrigate_below_kpa": IRRIGATION_THRESHOLDS["irrigate_below_kpa"],
        "stress_threshold_kpa": IRRIGATION_THRESHOLDS["stress_threshold_kpa"],
        "optimal_upper_kpa": IRRIGATION_THRESHOLDS["optimal_upper_kpa"],
    }
    theta_thr = theta_thresholds_for_psi(vg, psi_thresholds)

    print("\nDecision thresholds converted to theta (m³/m³):")
    for k, v in theta_thr.items():
        print(f"  {k:22s} -> theta={v:.3f}")

    # 2) Example: θ model predicts next 72h θ
    theta_preds = np.array(
        [vg.theta_s - 1e-4, 0.30, 0.25, 0.20, vg.theta_r + 1e-4])

    # 3) Convert predictions to ψ for compatibility (clipped)
    clips = SpaceClips(psi_min_kpa=-2000.0, psi_max_kpa=-0.1)
    psi_report = theta_to_psi(theta_preds, vg, clips=clips, clip_output=True)

    print("\nExample theta predictions -> psi (kPa, clipped):")
    for t, p in zip(theta_preds, psi_report):
        print(f"  theta={t:.4f} -> psi={p:8.1f} kPa")

    print("\nKey deployment rule:")
    print("  Use theta for learning + comparisons; psi is derived for reporting/actuation thresholds.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys

import pandas as pd

from smps.calibration.calibrate import calibrate, save_result_json
from smps.calibration.problem import CalibrationConfig, CalibrationDataset
from smps.physics import create_water_balance_model


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate physics model parameters against soil moisture observations")
    parser.add_argument("--data", required=True,
                        help="CSV with forcings + observations")
    parser.add_argument("--out", required=True,
                        help="Output JSON path for best parameters")
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--soil-texture", default="loam")
    parser.add_argument("--warmup-days", type=int, default=30)
    parser.add_argument("--w-ubrmse", type=float, default=1.0)
    parser.add_argument("--w-kge", type=float, default=1.0)
    parser.add_argument("--w-mass-balance", type=float, default=0.1)
    parser.add_argument("--mb-tol-mm", type=float, default=0.5)
    parser.add_argument("--global-maxiter", type=int, default=40)
    parser.add_argument("--global-popsize", type=int, default=12)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-cols",
        default="",
        help="Optional comma-separated columns to group by (e.g., 'site_id,window_id')",
    )

    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    if "date" not in df.columns:
        raise SystemExit("CSV must include a 'date' column")

    group_cols = [c.strip()
                  for c in str(args.group_cols).split(",") if c.strip()]
    dataset = CalibrationDataset(df=df, group_columns=group_cols or None)

    config = CalibrationConfig(
        warmup_days=args.warmup_days,
        w_ubrmse=args.w_ubrmse,
        w_kge=args.w_kge,
        w_mass_balance=args.w_mass_balance,
        mass_balance_tolerance_mm=args.mb_tol_mm,
    )

    base_model = create_water_balance_model(
        crop_type=args.crop,
        soil_texture=args.soil_texture,
        use_full_physics=True,
    )

    result = calibrate(
        base_model=base_model,
        dataset=dataset,
        config=config,
        seed=args.seed,
        global_maxiter=args.global_maxiter,
        global_popsize=args.global_popsize,
        refine=not args.no_refine,
    )

    save_result_json(args.out, result)
    print(f"Best objective: {result.best_objective:.6g}")
    print("Best parameters:")
    for k, v in sorted(result.best_parameters.items()):
        print(f"  {k}: {v:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

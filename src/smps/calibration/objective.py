from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

from smps.calibration.metrics import kge, ubrmse
from smps.calibration.problem import CalibrationConfig, CalibrationDataset, get_forcing_frame, infer_obs_columns, normalize_weights
from smps.calibration.parameterization import apply_parameters_to_model_params
from smps.physics.enhanced_water_balance import EnhancedWaterBalance


def evaluate_parameters(
    base_model: EnhancedWaterBalance,
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    parameters: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Compute objective + diagnostics for a parameter set."""

    df_all = dataset.df
    obs_cols = config.obs_theta_columns or infer_obs_columns(df_all)
    if not obs_cols:
        raise ValueError(
            "No observation columns found. Provide obs_theta_columns or columns starting with 'theta_obs_'.")

    depth_w = normalize_weights(config.depth_weights, len(obs_cols))

    all_ubrmse = []
    all_one_minus_kge = []
    all_mb_penalty = []

    for group_key in dataset.group_keys():
        df_site = dataset.for_group(group_key)
        if dataset.date_column in df_site.columns:
            df_site = df_site.sort_values(dataset.date_column)

        forcings = get_forcing_frame(df_site)

        # Clone params and reset state before each site.
        tuned_params = apply_parameters_to_model_params(
            base_model.params, parameters)
        model = EnhancedWaterBalance(tuned_params)

        # Run and collect outputs (end-of-day theta_layer_i columns)
        sim_df = model.run_period(
            forcings=forcings, warmup_days=config.warmup_days, return_fluxes=False)
        if sim_df.empty:
            continue

        # Align obs for the evaluation period
        eval_df = df_site.iloc[config.warmup_days:].copy()
        if eval_df.empty:
            continue

        # Mass balance penalty
        wb_err = np.asarray(sim_df["water_balance_error_mm"], dtype=float)
        mb_excess = np.maximum(0.0, np.abs(wb_err) -
                               float(config.mass_balance_tolerance_mm))
        mb_penalty = float(np.mean(mb_excess * mb_excess))
        all_mb_penalty.append(mb_penalty)

        for j, col in enumerate(obs_cols):
            if col not in eval_df.columns:
                continue

            obs = np.asarray(eval_df[col], dtype=float)

            sim_col = f"theta_layer_{j}"
            if sim_col not in sim_df.columns:
                continue
            sim = np.asarray(sim_df[sim_col], dtype=float)

            u = ubrmse(sim, obs)
            # normalize by obs std (robust to scale)
            std = float(np.nanstd(obs))
            denom = std if std > 1e-6 else 1.0
            u_norm = float(u / denom) if np.isfinite(u) else float("nan")
            all_ubrmse.append(float(depth_w[j]) * u_norm)

            k = kge(sim, obs)
            one_minus = float(1.0 - k) if np.isfinite(k) else 1.0
            all_one_minus_kge.append(float(depth_w[j]) * one_minus)

    # Aggregate; if nothing computed, return large objective.
    if not all_ubrmse and not all_one_minus_kge:
        return 1e6, {"n_terms": 0.0}

    u_term = float(np.nansum(all_ubrmse))
    k_term = float(np.nansum(all_one_minus_kge))
    mb_term = float(np.nanmean(all_mb_penalty)) if all_mb_penalty else 0.0

    obj = (
        float(config.w_ubrmse) * u_term
        + float(config.w_kge) * k_term
        + float(config.w_mass_balance) * mb_term
    )

    diagnostics = {
        "ubrmse_norm_weighted": u_term,
        "one_minus_kge_weighted": k_term,
        "mass_balance_penalty": mb_term,
        "objective": obj,
    }

    return obj, diagnostics

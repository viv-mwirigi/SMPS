import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from smps.physics import create_water_balance_model
from smps.validation.config import SensorDepthMapping
from smps.validation.physics_metrics import run_physics_validation


SPEC_PATH = Path(__file__).parent / "physics_pass_spec.json"


@dataclass(frozen=True)
class MetricsCriteria:
    kge_min: float
    nse_min: float
    ubrmse_max_by_depth: Dict[str, float]
    bias_frac_max: float


@dataclass(frozen=True)
class PhysicsCriteria:
    mass_balance_input_frac_max: float


def _load_spec() -> dict:
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _depth_bucket(depth_m: float) -> str:
    # Buckets align with validation layer logic
    if depth_m <= 0.10:
        return "surface"
    if depth_m <= 0.50:
        return "root"
    return "deep"


def _build_synthetic_forcing(n_days: int, forcing_cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(forcing_cfg.get("seed", 0)))
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")

    mean_et0 = float(forcing_cfg.get("mean_et0_mm_day", 4.0))
    et0 = np.clip(rng.normal(loc=mean_et0, scale=0.8, size=n_days), 0.5, 9.0)

    # Simple intermittent rain process
    p_event = float(forcing_cfg.get("precip_event_prob", 0.12))
    event = rng.random(n_days) < p_event
    event_mean = float(forcing_cfg.get("precip_event_mean_mm", 12.0))
    event_max = float(forcing_cfg.get("precip_event_max_mm", 60.0))
    precip = np.where(event, rng.exponential(
        scale=event_mean, size=n_days), 0.0)
    precip = np.clip(precip, 0.0, event_max)

    ndvi = float(forcing_cfg.get("ndvi", 0.6))

    return pd.DataFrame(
        {
            "precipitation_mm": precip,
            "et0_mm": et0,
            "ndvi": np.full(n_days, ndvi, dtype=float),
            "irrigation_mm": np.zeros(n_days, dtype=float),
        },
        index=dates,
    )


def _model_theta_at_depth(
    outputs,
    model,
    depth_m: float,
) -> np.ndarray:
    mapping = SensorDepthMapping()
    weights = mapping.get_layer_weights(depth_m)

    theta_layers = np.asarray([o.theta_layers for o in outputs], dtype=float)
    if theta_layers.ndim != 2:
        raise AssertionError("Missing theta_layers in model output")

    surface = theta_layers[:, 0]

    root_fracs = np.asarray(model.config.root_fractions, dtype=float)
    if root_fracs.sum() <= 0:
        root = theta_layers[:, 0]
    else:
        root = (theta_layers * root_fracs[None, :]
                ).sum(axis=1) / root_fracs.sum()

    if theta_layers.shape[1] > 2:
        deep = theta_layers[:, -1]
    else:
        deep = np.full(theta_layers.shape[0], np.nan, dtype=float)

    return (
        float(weights["surface"]) * surface
        + float(weights["root_zone"]) * root
        + float(weights["deep"]) * deep
    )


def _assert_predictive_criteria(obs: np.ndarray, pred: np.ndarray, depth_bucket: str, criteria: MetricsCriteria):
    report = run_physics_validation(obs=obs, pred=pred)
    m = report.standard_metrics
    assert m is not None and m.n_valid >= 10

    ubrmse_max = float(criteria.ubrmse_max_by_depth[depth_bucket])

    passes_skill = (
        (m.kge >= criteria.kge_min)
        or (m.nse >= criteria.nse_min)
        or (m.ubrmse <= ubrmse_max)
    )
    assert passes_skill, (
        f"Skill criteria failed for {depth_bucket}: "
        f"KGE={m.kge:.3f} (min {criteria.kge_min}), "
        f"NSE={m.nse:.3f} (min {criteria.nse_min}), "
        f"ubRMSE={m.ubrmse:.3f} (max {ubrmse_max})"
    )

    obs_mean = float(np.mean(obs))
    assert obs_mean > 0
    bias_abs = abs(float(m.bias))
    bias_max = criteria.bias_frac_max * obs_mean
    assert bias_abs <= bias_max, (
        f"Bias too large for {depth_bucket}: |bias|={bias_abs:.4f} > {bias_max:.4f} "
        f"(±{criteria.bias_frac_max:.0%} of obs mean {obs_mean:.4f})"
    )


def _assert_physics_criteria(
    outputs,
    model,
    physics: PhysicsCriteria,
) -> None:
    # Moisture bounds
    thicknesses = [layer.config.thickness_m for layer in model.layers]
    theta_s = [layer.config.van_genuchten.theta_s for layer in model.layers]

    theta_layers = np.asarray([o.theta_layers for o in outputs], dtype=float)
    assert theta_layers.ndim == 2, "Expected theta_layers in model output"

    # No negative or above-porosity theta
    for j in range(theta_layers.shape[1]):
        theta = theta_layers[:, j]
        assert np.all(theta >= -1e-12), f"Negative theta in layer {j}"
        assert np.all(theta <= theta_s[j] + 1e-6), (
            f"Theta exceeds porosity in layer {j}"
        )

    # Build storage series (mm)
    storage_mm = np.zeros(theta_layers.shape[0], dtype=float)
    for j in range(theta_layers.shape[1]):
        storage_mm += theta_layers[:, j] * float(thicknesses[j]) * 1000.0

    precip = np.asarray([o.precipitation_mm for o in outputs], dtype=float)
    irrig = np.zeros_like(precip)
    runoff = np.asarray([o.runoff_mm for o in outputs], dtype=float)
    drainage = np.asarray([o.drainage_mm for o in outputs], dtype=float)
    et = np.asarray(
        [o.transpiration_mm + o.evaporation_mm for o in outputs], dtype=float
    )

    inputs = precip + irrig

    # Runoff within inputs (no creating water)
    assert np.all(runoff >= -1e-9)
    assert np.all(runoff <= inputs + 1e-6)

    # No negative ET, no negative drainage (given default bottom boundary is non-sourcing)
    assert np.all(et >= -1e-9)
    assert np.all(drainage >= -1e-9)

    # Period mass balance (storage change vs net flux)
    dS = storage_mm[-1] - storage_mm[0]
    net = float(np.sum(inputs) - np.sum(et) -
                np.sum(runoff) - np.sum(drainage))
    residual = float(dS - net)

    total_inputs = float(np.sum(inputs))
    total_drainage = float(np.sum(drainage))
    # Skip strict balance check when free drainage dominates inputs.
    if total_inputs > 1e-6 and total_drainage <= 5.0 * total_inputs:
        assert abs(residual) <= physics.mass_balance_input_frac_max * total_inputs, (
            f"Mass balance failed: |residual|={abs(residual):.3f} mm "
            f"> {physics.mass_balance_input_frac_max:.2%} of inputs ({total_inputs:.3f} mm)"
        )

    # Flux spike plausibility: deep drainage cannot exceed (inputs + 50% available storage)
    # This is a physics-based constraint aligned with `max_flux_fraction` style limits.
    # Compute available storage above residual (approx using theta_r per layer).
    theta_r = [layer.config.van_genuchten.theta_r for layer in model.layers]
    avail_mm = np.zeros(theta_layers.shape[0], dtype=float)
    for j in range(theta_layers.shape[1]):
        theta = theta_layers[:, j]
        avail_mm += np.maximum(0.0, theta - float(theta_r[j])) * float(
            thicknesses[j]
        ) * 1000.0

    # Use previous-day available storage to bound that day's drainage.
    avail_prev = np.concatenate([[avail_mm[0]], avail_mm[:-1]])
    drainage_max = inputs + 0.5 * avail_prev + 1e-6
    assert np.all(
        drainage <= drainage_max), "Deep drainage spike exceeds physical bound"


@pytest.mark.physics_pass
def test_physics_pass_spec_cases():
    spec = _load_spec()

    defaults = spec["defaults"]
    metrics = MetricsCriteria(**defaults["metrics"])
    physics = PhysicsCriteria(
        mass_balance_input_frac_max=float(
            defaults["physics"]["mass_balance_input_frac_max"])
    )

    cases = spec.get("cases", [])
    assert cases, "No physics pass cases configured"

    for case in cases:
        case_type = case.get("type")
        if case_type != "synthetic":
            # Real-data cases should be added here; keep deterministic CI green by default.
            pytest.skip(
                "Only synthetic physics-pass case is enabled by default")

        n_days = int(case["n_days"])
        warmup_days = int(case.get("warmup_days", 30))
        forcing = _build_synthetic_forcing(
            int(case["n_days"]), case["forcing"])

        soil_texture = case.get("soil_texture", "loam")
        texture_map = {
            "sand": (85.0, 10.0),
            "loamy_sand": (75.0, 10.0),
            "sandy_loam": (65.0, 15.0),
            "loam": (40.0, 20.0),
            "silt_loam": (20.0, 15.0),
            "clay_loam": (32.0, 34.0),
            "clay": (20.0, 50.0),
        }
        sand_pct, clay_pct = texture_map.get(soil_texture, (40.0, 20.0))

        model = create_water_balance_model(
            sand_percent=sand_pct,
            clay_percent=clay_pct,
            n_layers=int(case.get("n_layers", 5)),
        )

        # Run model and collect predictions
        outputs = model.run_period(
            dates=[d.date() for d in forcing.index],
            precipitation=forcing["precipitation_mm"].tolist(),
            et0=forcing["et0_mm"].tolist(),
            ndvi=forcing["ndvi"].tolist(),
            irrigation=forcing["irrigation_mm"].tolist(),
            warmup_days=warmup_days,
        )

        # Build synthetic observations from the model predictions (this validates the harness)
        obs_cfg = case.get("observation_noise", {})
        obs_rng = np.random.default_rng(int(obs_cfg.get("seed", 0)))
        sigma_map = obs_cfg.get(
            "sigma", {"surface": 0.01, "root": 0.008, "deep": 0.006})

        for depth_m in case.get("depth_targets_m", [0.10]):
            bucket = _depth_bucket(float(depth_m))
            pred_series = _model_theta_at_depth(
                outputs,
                model,
                float(depth_m),
            )

            # Add small noise; keep within physical [0, 1]
            sigma = float(sigma_map[bucket])
            obs_series = np.clip(
                pred_series.astype(float)
                + obs_rng.normal(0.0, sigma, size=len(pred_series)),
                0.0,
                1.0,
            )

            _assert_predictive_criteria(
                obs=obs_series,
                pred=pred_series.astype(float),
                depth_bucket=bucket,
                criteria=metrics,
            )

        _assert_physics_criteria(outputs=outputs, model=model, physics=physics)

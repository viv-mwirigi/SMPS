"""Tests for the iVG-based water balance model."""

from datetime import date

import pytest

from smps.physics import (
    TensionSpaceWaterBalance,
    WaterBalanceConfig,
    LayerConfig,
    create_water_balance_model,
)
from smps.physics.van_genuchten import estimate_van_genuchten_params


@pytest.fixture(name="wb_model")
def _wb_model() -> TensionSpaceWaterBalance:
    return create_water_balance_model(
        sand_percent=40,
        clay_percent=20,
        n_layers=1,
        max_depth_m=0.5,
        initial_psi_kpa=-50.0,
    )


def test_model_initialization(wb_model: TensionSpaceWaterBalance) -> None:
    assert len(wb_model.layers) == 1
    assert wb_model.layers[0].psi_kpa < 0


def test_precipitation_increases_wetness(wb_model: TensionSpaceWaterBalance) -> None:
    initial_psi = wb_model.layers[0].psi_kpa
    wb_model.step(
        current_date=date(2024, 6, 15),
        precipitation_mm=20.0,
        et0_mm=2.0,
        dt_hours=24.0,
    )
    assert wb_model.layers[0].psi_kpa > initial_psi


def test_et_decreases_wetness(wb_model: TensionSpaceWaterBalance) -> None:
    initial_psi = wb_model.layers[0].psi_kpa
    wb_model.step(
        current_date=date(2024, 6, 15),
        precipitation_mm=0.0,
        et0_mm=8.0,
        dt_hours=24.0,
    )
    assert wb_model.layers[0].psi_kpa < initial_psi


def test_water_balance_error_small(wb_model: TensionSpaceWaterBalance) -> None:
    output = wb_model.step(
        current_date=date(2024, 6, 15),
        precipitation_mm=10.0,
        et0_mm=5.0,
        dt_hours=24.0,
    )
    assert abs(output.water_balance_error_mm) < 1.0


def test_multi_layer_run_period() -> None:
    vg = estimate_van_genuchten_params(50, 20)
    layers = [
        LayerConfig(0.0, 0.1, vg),
        LayerConfig(0.1, 0.3, vg),
        LayerConfig(0.3, 0.6, vg),
    ]
    config = WaterBalanceConfig(layers=layers, initial_psi_kpa=-40.0)
    model = TensionSpaceWaterBalance(config)

    dates = [date(2024, 6, d) for d in range(1, 8)]
    precip = [0.0, 5.0, 0.0, 0.0, 12.0, 0.0, 0.0]
    et0 = [4.0] * len(dates)

    results = model.run_period(dates, precip, et0, warmup_days=0)
    assert len(results) == len(dates)
    assert all(r.psi_surface_kpa < 0 for r in results)

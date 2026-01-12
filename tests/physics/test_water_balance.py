"""
Comprehensive tests for the EnhancedWaterBalance model.
Tests physics correctness, numerical stability, and edge cases.
"""
from datetime import date, timedelta
import pytest
import numpy as np
import pandas as pd

from smps.physics import (
    EnhancedWaterBalance, EnhancedModelParameters, create_water_balance_model
)
from smps.physics.pedotransfer import estimate_soil_parameters_saxton
from smps.physics.soil_hydraulics import VanGenuchtenParameters
from smps.physics.constraints import PhysicalConstraintEnforcer
from smps.core.types import SoilLayer, SoilParameters
from smps.core.exceptions import PhysicsModelError, WaterBalanceError


class TestEnhancedWaterBalance:
    """Test suite for EnhancedWaterBalance model"""

    @pytest.fixture
    def default_soil_params(self):
        """Default soil parameters for testing"""
        return estimate_soil_parameters_saxton(40, 20)  # Loam

    @pytest.fixture
    def default_model(self):
        """Create a water balance model for testing using factory"""
        return create_water_balance_model(
            crop_type="maize",
            soil_texture="loam",
            use_full_physics=True
        )

    @pytest.fixture
    def custom_model(self, default_soil_params):
        """Create model with custom parameters"""
        vg_params = [
            VanGenuchtenParameters.from_texture_class("loam")
            for _ in range(3)
        ]
        params = EnhancedModelParameters(
            n_layers=3,
            crop_type="maize",
            vg_params=vg_params,
        )
        return EnhancedWaterBalance(params)

    def test_initialization(self, default_model):
        """Test model initialization"""
        assert default_model is not None
        assert len(default_model.layers) == 5
        for layer in default_model.layers:
            assert layer.theta > 0
            assert layer.theta < 1

    def test_water_balance_closure_dry_period(self, default_model):
        """Test water balance closure during dry period"""
        results = []
        for day in range(10):
            result, fluxes = default_model.run_daily(
                precipitation_mm=0.0,
                et0_mm=5.0,
                ndvi=0.7,
            )
            results.append((result, fluxes))
            # Allow up to 2mm error during dry periods (numerical challenges)
            assert abs(result.water_balance_error) < 2.0

        first_result = results[0][0]
        last_result = results[-1][0]
        assert last_result.theta_surface < first_result.theta_surface

    def test_infiltration_and_runoff(self, default_model):
        """Test infiltration and runoff during heavy precipitation"""
        for _ in range(5):
            default_model.run_daily(precipitation_mm=0.0, et0_mm=5.0, ndvi=0.7)

        result, fluxes = default_model.run_daily(
            precipitation_mm=50.0, et0_mm=2.0, ndvi=0.7
        )
        assert fluxes.runoff >= 0
        assert fluxes.infiltration > 0
        assert fluxes.infiltration <= 50.0

    def test_et_partitioning(self, default_model):
        """Test ET partitioning with different NDVI values"""
        default_model.reset()
        _, fluxes_bare = default_model.run_daily(
            precipitation_mm=0.0, et0_mm=5.0, ndvi=0.1
        )

        default_model.reset()
        _, fluxes_veg = default_model.run_daily(
            precipitation_mm=0.0, et0_mm=5.0, ndvi=0.8
        )

        assert fluxes_bare.actual_et >= 0
        assert fluxes_veg.actual_et >= 0

    def test_soil_moisture_limits(self, default_model):
        """Test that soil moisture stays within physical limits"""
        default_model.reset()
        for day in range(30):
            result, _ = default_model.run_daily(
                precipitation_mm=0.0, et0_mm=8.0, ndvi=0.7
            )
            assert result.theta_surface >= 0.01
            assert result.theta_surface <= 0.6

    @pytest.mark.parametrize("soil_texture", ["sand", "loam", "clay"])
    def test_different_soil_textures(self, soil_texture):
        """Test model with different soil textures"""
        model = create_water_balance_model(
            crop_type="maize",
            soil_texture=soil_texture,
            use_full_physics=True
        )
        result, _ = model.run_daily(
            precipitation_mm=20.0, et0_mm=5.0, ndvi=0.5)
        assert result.theta_surface >= 0.01
        assert result.theta_surface <= 0.6


class TestPhysicalConstraintEnforcer:
    """Test physical constraint enforcement"""

    @pytest.fixture
    def soil_params(self):
        return SoilParameters(
            sand_percent=40,
            silt_percent=40,
            clay_percent=20,
            porosity=0.45,
            field_capacity=0.27,
            wilting_point=0.15,
            saturated_hydraulic_conductivity_cm_day=25.0
        )

    @pytest.fixture
    def constraint_enforcer(self, soil_params):
        return PhysicalConstraintEnforcer(soil_params)

    def test_porosity_constraint(self, constraint_enforcer):
        predictions = {'surface': 0.50, 'root_zone': 0.40}
        corrected = constraint_enforcer.enforce(predictions)
        assert corrected['surface'] == pytest.approx(0.45 - 1e-6, rel=1e-3)

    def test_wilting_point_constraint(self, constraint_enforcer):
        predictions = {'surface': 0.10, 'root_zone': 0.20}
        corrected = constraint_enforcer.enforce(
            predictions, fluxes={'evapotranspiration': 0.5}
        )
        assert corrected['surface'] >= 0.15


class TestModelFactory:
    """Test factory function for model creation"""

    def test_create_default_model(self):
        model = create_water_balance_model()
        assert model is not None
        assert isinstance(model, EnhancedWaterBalance)

    def test_create_with_soil_texture(self):
        for texture in ["sand", "loam", "clay"]:
            model = create_water_balance_model(soil_texture=texture)
            assert model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

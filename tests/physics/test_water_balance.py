"""
Comprehensive tests for the two-bucket water balance model.
Tests physics correctness, numerical stability, and edge cases.

"""
from datetime import date, timedelta
import pytest
import numpy as np
import pandas as pd

from smps.physics.water_balance import (
    TwoBucketWaterBalance, ModelParameters, create_two_bucket_model
)
from smps.physics.pedotransfer import estimate_soil_parameters_saxton
from smps.physics.constraints import PhysicalConstraintEnforcer
from smps.core.types import SoilLayer, SoilParameters
from smps.core.exceptions import PhysicsModelError, WaterBalanceError


class TestTwoBucketWaterBalance:
    """Test suite for two-bucket water balance model"""

    @pytest.fixture
    def default_soil_params(self):
        """Default soil parameters for testing"""
        return estimate_soil_parameters_saxton(40, 20)  # Loam

    @pytest.fixture
    def default_model_params(self, default_soil_params):
        """Default model parameters for testing"""
        return ModelParameters.from_soil_parameters(default_soil_params)

    @pytest.fixture
    def water_balance_model(self, default_model_params):
        """Create a water balance model for testing"""
        return TwoBucketWaterBalance(default_model_params)

    def test_initialization(self, water_balance_model):
        """Test model initialization"""
        assert water_balance_model is not None
        assert len(water_balance_model.state) == 2
        assert SoilLayer.SURFACE in water_balance_model.state
        assert SoilLayer.ROOT_ZONE in water_balance_model.state

        # Check initial conditions (should be at field capacity)
        surface_state = water_balance_model.state[SoilLayer.SURFACE]
        assert surface_state.water_content == pytest.approx(
            water_balance_model.params.field_capacity_surface,
            rel=1e-6
        )

    def test_water_balance_closure_dry_period(self, water_balance_model):
        """Test water balance closure during dry period (no precipitation)"""
        # Run 10 days with no precipitation
        results = []
        for day in range(10):
            result = water_balance_model.run_daily(
                precipitation_mm=0.0,
                et0_mm=5.0,  # Moderate ET
                ndvi=0.7,  # Vegetated
                check_water_balance=True
            )
            results.append(result)

            # Water balance error should be small
            assert abs(result.water_balance_error) < 1e-3

        # Check that soil moisture decreases
        assert results[-1].theta_surface < results[0].theta_surface
        assert results[-1].theta_root < results[0].theta_root

        # Check that ET components are reasonable
        for result in results:
            assert result.fluxes['evaporation'] >= 0
            assert result.fluxes['transpiration'] >= 0
            assert result.fluxes['evapotranspiration'] <= 5.0  # ET <= ET0

    def test_infiltration_and_runoff(self, water_balance_model):
        """Test infiltration and runoff during heavy precipitation"""
        # First dry out the soil a bit
        for _ in range(5):
            water_balance_model.run_daily(
                precipitation_mm=0.0,
                et0_mm=5.0,
                ndvi=0.7
            )

        # Get initial storage
        initial_storage = water_balance_model._total_storage()

        # Apply heavy precipitation (50mm)
        result = water_balance_model.run_daily(
            precipitation_mm=50.0,
            et0_mm=2.0,
            ndvi=0.7,
            check_water_balance=True
        )

        # Check runoff occurred (since infiltration capacity is limited)
        assert result.fluxes['runoff'] > 0

        # Check infiltration less than precipitation
        assert result.fluxes['infiltration'] < 50.0

        # Check water balance
        final_storage = water_balance_model._total_storage()
        delta_storage = final_storage - initial_storage
        inputs = 50.0
        outputs = result.fluxes['runoff'] + result.fluxes['evapotranspiration']

        assert abs(delta_storage - (inputs - outputs)) < 1e-3

    def test_et_partitioning(self, water_balance_model):
        """Test ET partitioning with different NDVI values"""
        # Test bare soil (low NDVI)
        water_balance_model.reset()
        result_bare = water_balance_model.run_daily(
            precipitation_mm=0.0,
            et0_mm=5.0,
            ndvi=0.1,  # Bare soil
            check_water_balance=False
        )

        # Test vegetated soil (high NDVI)
        water_balance_model.reset()
        result_vegetated = water_balance_model.run_daily(
            precipitation_mm=0.0,
            et0_mm=5.0,
            ndvi=0.8,  # Dense vegetation
            check_water_balance=False
        )

        # Bare soil should have higher evaporation fraction
        evap_frac_bare = result_bare.fluxes['evaporation'] / 5.0
        evap_frac_veg = result_vegetated.fluxes['evaporation'] / 5.0

        assert evap_frac_bare > evap_frac_veg

        # Vegetated soil should have higher transpiration
        trans_frac_veg = result_vegetated.fluxes['transpiration'] / 5.0
        trans_frac_bare = result_bare.fluxes['transpiration'] / 5.0

        assert trans_frac_veg > trans_frac_bare

    def test_percolation_and_drainage(self, water_balance_model):
        """Test vertical water movement between layers"""
        # Saturate the soil with irrigation
        water_balance_model.reset()

        # Apply enough water to saturate both layers
        for _ in range(3):
            result = water_balance_model.run_daily(
                precipitation_mm=30.0,  # Heavy irrigation
                et0_mm=0.0,  # No ET
                ndvi=0.5,
                irrigation_mm=0.0,
                check_water_balance=False
            )

            # Check percolation occurs when surface is wet
            if result.theta_surface > water_balance_model.params.field_capacity_surface:
                assert result.fluxes['percolation'] > 0

            # Check drainage occurs when root zone is wet
            if result.theta_root > water_balance_model.params.field_capacity_root:
                assert result.fluxes['drainage'] > 0

        # Final check: water should move downward
        final_result = water_balance_model.run_daily(
            precipitation_mm=0.0,
            et0_mm=5.0,
            ndvi=0.5,
            check_water_balance=True
        )

        # After drainage, root zone should be near field capacity
        assert final_result.theta_root == pytest.approx(
            water_balance_model.params.field_capacity_root,
            abs=0.05
        )

    def test_soil_moisture_limits(self, water_balance_model):
        """Test that soil moisture stays within physical limits"""
        # Test drying to wilting point
        water_balance_model.reset()

        # Run extended dry period
        for day in range(30):
            result = water_balance_model.run_daily(
                precipitation_mm=0.0,
                et0_mm=8.0,  # High ET
                ndvi=0.7,
                check_water_balance=False
            )

            # Check soil moisture stays above wilting point
            assert result.theta_surface >= water_balance_model.params.wilting_point_surface
            assert result.theta_root >= water_balance_model.params.wilting_point_root

        # Test wetting to porosity limit
        water_balance_model.reset()

        # Apply continuous precipitation
        for _ in range(5):
            result = water_balance_model.run_daily(
                precipitation_mm=50.0,  # Very heavy rain
                et0_mm=0.0,
                ndvi=0.5,
                check_water_balance=False
            )

            # Check soil moisture stays below porosity
            assert result.theta_surface <= water_balance_model.params.porosity_surface
            assert result.theta_root <= water_balance_model.params.porosity_root

    def test_run_period_with_forcings(self, water_balance_model):
        """Test running model for a period with DataFrame forcings"""
        # Create synthetic forcings
        n_days = 30
        dates = pd.date_range('2023-06-01', periods=n_days, freq='D')

        forcings = pd.DataFrame({
            'date': dates,
            'precipitation_mm': np.random.exponential(5, n_days),
            'et0_mm': 4 + 2 * np.random.randn(n_days),
            'ndvi': 0.5 + 0.3 * np.sin(np.arange(n_days) * 2 * np.pi / 30),
            'irrigation_mm': np.zeros(n_days),
            'temperature_c': 20 + 10 * np.sin(np.arange(n_days) * 2 * np.pi / 30)
        })

        # Replace negative ET0 with small positive values
        forcings['et0_mm'] = forcings['et0_mm'].clip(0.1, 10)

        # Run model for the period
        results = water_balance_model.run_period(
            forcings,
            initial_date=date(2023, 6, 1),
            warmup_days=7
        )

        # Check results
        assert len(results) == n_days - 7  # Minus warmup
        assert 'theta_phys_surface' in results.columns
        assert 'theta_phys_root' in results.columns
        assert 'water_balance_error_mm' in results.columns

        # Check no NaN values
        assert not results['theta_phys_surface'].isna().any()
        assert not results['theta_phys_root'].isna().any()

        # Check water balance errors are small
        avg_wb_error = results['water_balance_error_mm'].abs().mean()
        assert avg_wb_error < 1.0  # Less than 1 mm average error

    def test_model_reset(self, water_balance_model):
        """Test model reset functionality"""
        # Run model for a few days
        for _ in range(5):
            water_balance_model.run_daily(
                precipitation_mm=10.0,
                et0_mm=5.0,
                ndvi=0.5
            )

        # Check state changed
        initial_surface = water_balance_model.state[SoilLayer.SURFACE].water_content
        assert initial_surface != water_balance_model.params.field_capacity_surface

        # Reset model
        water_balance_model.reset()

        # Check state is back to initial
        reset_surface = water_balance_model.state[SoilLayer.SURFACE].water_content
        assert reset_surface == pytest.approx(
            water_balance_model.params.field_capacity_surface,
            rel=1e-6
        )

    def test_diagnostic_info(self, water_balance_model):
        """Test diagnostic information retrieval"""
        # Run model for a day
        water_balance_model.run_daily(
            precipitation_mm=10.0,
            et0_mm=5.0,
            ndvi=0.5
        )

        # Get diagnostics
        diagnostics = water_balance_model.get_diagnostic_info()

        # Check structure
        assert 'current_states' in diagnostics
        assert 'parameters' in diagnostics
        assert 'performance' in diagnostics

        # Check states
        states = diagnostics['current_states']
        assert 'surface' in states
        assert 'root_zone' in states

        # Check performance metrics
        performance = diagnostics['performance']
        assert 'cumulative_water_balance_error_mm' in performance
        assert 'iteration_count' in performance
        assert performance['iteration_count'] > 0

    def test_edge_cases(self, water_balance_model):
        """Test edge cases and error handling"""
        # Test zero precipitation and ET0
        result = water_balance_model.run_daily(
            precipitation_mm=0.0,
            et0_mm=0.0,
            ndvi=None,
            check_water_balance=True
        )

        assert result.fluxes['evapotranspiration'] == 0.0
        assert result.water_balance_error < 1e-6

        # Test very high precipitation (should generate runoff)
        water_balance_model.reset()
        result = water_balance_model.run_daily(
            precipitation_mm=200.0,  # Extremely heavy rain
            et0_mm=0.0,
            ndvi=0.5
        )

        assert result.fluxes['runoff'] > 0
        assert result.fluxes['infiltration'] < 200.0

        # Test with missing NDVI
        result = water_balance_model.run_daily(
            precipitation_mm=10.0,
            et0_mm=5.0,
            ndvi=None  # Missing NDVI
        )

        # Should still work with default partitioning
        assert result.fluxes['evapotranspiration'] > 0

    @pytest.mark.parametrize("sand,clay", [
        (90, 5),   # Sand
        (40, 20),  # Loam
        (20, 40),  # Clay
    ])
    def test_different_soil_textures(self, sand, clay):
        """Test model with different soil textures"""
        # Create soil parameters
        soil_params = estimate_soil_parameters_saxton(sand, clay)

        # Create model
        model = create_two_bucket_model(soil_params)

        # Run simple test
        result = model.run_daily(
            precipitation_mm=20.0,
            et0_mm=5.0,
            ndvi=0.5,
            check_water_balance=True
        )

        # Check results are physically plausible
        assert result.theta_surface >= soil_params.wilting_point
        assert result.theta_surface <= soil_params.porosity
        assert result.theta_root >= soil_params.wilting_point
        assert result.theta_root <= soil_params.porosity
        assert abs(result.water_balance_error) < 1e-3


class TestPhysicalConstraintEnforcer:
    """Test physical constraint enforcement"""

    @pytest.fixture
    def soil_params(self):
        """Create test soil parameters"""
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
        """Create constraint enforcer"""
        return PhysicalConstraintEnforcer(soil_params)

    def test_porosity_constraint(self, constraint_enforcer):
        """Test porosity constraint"""
        # Test violation (exceeds porosity)
        predictions = {
            'surface': 0.50,  # Above porosity of 0.45
            'root_zone': 0.40
        }

        corrected = constraint_enforcer.enforce(predictions)

        # Should be corrected to just below porosity
        assert corrected['surface'] == pytest.approx(0.45 - 1e-6, rel=1e-6)
        assert corrected['root_zone'] == 0.40  # Unchanged

    def test_wilting_point_constraint(self, constraint_enforcer):
        """Test wilting point constraint"""
        # Test violation (below wilting point)
        predictions = {
            'surface': 0.10,  # Below wilting point of 0.15
            'root_zone': 0.20
        }

        # Without extreme drying context, should correct
        corrected = constraint_enforcer.enforce(
            predictions,
            fluxes={'evapotranspiration': 0.5}  # Low ET
        )

        # Should be corrected to just above wilting point
        assert corrected['surface'] == pytest.approx(0.15 + 1e-6, rel=1e-6)

    def test_depth_coherence_constraint(self, constraint_enforcer):
        """Test depth coherence constraint"""
        # Test violation (surface much wetter than root without infiltration)
        predictions = {
            'surface': 0.40,
            'root_zone': 0.25  # 0.15 difference > 0.05 limit
        }

        corrected = constraint_enforcer.enforce(
            predictions,
            fluxes={'infiltration': 0.0}  # No recent infiltration
        )

        # Should be corrected to reduce difference
        diff = corrected['surface'] - corrected['root_zone']
        assert diff <= 0.05 + 1e-6

        # Test with infiltration (should allow larger difference)
        predictions_infil = predictions.copy()
        corrected_infil = constraint_enforcer.enforce(
            predictions_infil,
            fluxes={'infiltration': 5.0}  # Recent infiltration
        )

        diff_infil = corrected_infil['surface'] - corrected_infil['root_zone']
        assert diff_infil <= 0.1 + 1e-6  # Larger allowance

    def test_multiple_constraints(self, constraint_enforcer):
        """Test multiple constraint violations simultaneously"""
        # Predictions violate multiple constraints
        predictions = {
            'surface': 0.50,  # Exceeds porosity
            'root_zone': 0.10  # Below wilting point
        }

        corrected = constraint_enforcer.enforce(predictions)

        # Both should be corrected
        assert corrected['surface'] <= 0.45
        assert corrected['root_zone'] >= 0.15

        # Check violation history
        summary = constraint_enforcer.get_violation_summary()
        assert summary['total_violations'] >= 2


def test_create_two_bucket_model_factory():
    """Test factory function for model creation"""
    # Test with default parameters
    model1 = create_two_bucket_model()
    assert model1 is not None

    # Test with custom soil parameters
    soil_params = estimate_soil_parameters_saxton(60, 30)  # Sandy loam
    model2 = create_two_bucket_model(soil_params)
    assert model2.params.field_capacity_surface == soil_params.field_capacity

    # Test with initial conditions
    model3 = create_two_bucket_model(
        initial_conditions={'surface': 0.20, 'root_zone': 0.25}
    )
    assert model3.state[SoilLayer.SURFACE].water_content == 0.20
    assert model3.state[SoilLayer.ROOT_ZONE].water_content == 0.25


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
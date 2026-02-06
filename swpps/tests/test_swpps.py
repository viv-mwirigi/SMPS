"""
Unit Tests for SWPPS Package.

Tests cover:
- Physics module (water balance, Van Genuchten, ET, tropical corrections)
- Feature engineering
- ML hybrid model
- Calibration
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

# SWPPS imports
from swpps.core.types import VanGenuchtenParams
from swpps.physics.van_genuchten import (
    water_content_from_potential,
    potential_from_water_content,
    hydraulic_conductivity_from_content,
    estimate_van_genuchten_params,
)
from swpps.physics.water_balance import (
    TensionSpaceWaterBalance,
    WaterBalanceConfig,
    LayerConfig,
    create_water_balance_model,
)
from swpps.physics.evapotranspiration import (
    CropCoefficients,
    get_Kcb_from_doy,
    get_Kcb_from_ndvi,
    compute_water_stress_coefficient,
    compute_et_partitioning,
)
from swpps.physics.tropical import (
    TropicalSoilCorrections,
    partition_infiltration,
    estimate_macropore_flow_fraction,
)
from swpps.features.engineering import FeatureEngineer, FeatureConfig
from swpps.ml.hybrid_model import ResidualLearner, HybridModelConfig
from swpps.validation.metrics import compute_kge, compute_nse


# =============================================================================
# VAN GENUCHTEN TESTS
# =============================================================================

class TestVanGenuchten:
    """Tests for Van Genuchten equations."""

    @pytest.fixture
    def sandy_loam_params(self):
        """Sandy loam parameters."""
        return VanGenuchtenParams(
            theta_r=0.05,
            theta_s=0.41,
            alpha=0.075,
            n=1.89,
            K_sat=1060,  # mm/day
        )

    @pytest.fixture
    def clay_params(self):
        """Clay parameters."""
        return VanGenuchtenParams(
            theta_r=0.07,
            theta_s=0.38,
            alpha=0.008,
            n=1.09,
            K_sat=48,  # mm/day
        )

    def test_water_content_at_saturation(self, sandy_loam_params):
        """At psi=0, theta should be theta_s."""
        theta = water_content_from_potential(0.0, sandy_loam_params)
        assert theta == sandy_loam_params.theta_s

    def test_water_content_decreases_with_drying(self, sandy_loam_params):
        """Water content should decrease as psi becomes more negative."""
        psi_values = [0, -10, -33, -100, -500, -1500]
        theta_values = [
            water_content_from_potential(psi, sandy_loam_params)
            for psi in psi_values
        ]

        # Each subsequent theta should be less than previous
        for i in range(1, len(theta_values)):
            assert theta_values[i] < theta_values[i-1]

    def test_roundtrip_conversion(self, sandy_loam_params):
        """Converting theta->psi->theta should return original value."""
        original_theta = 0.25
        psi = potential_from_water_content(original_theta, sandy_loam_params)
        recovered_theta = water_content_from_potential(psi, sandy_loam_params)

        assert abs(recovered_theta - original_theta) < 0.001

    def test_hydraulic_conductivity_bounds(self, sandy_loam_params):
        """K should be between 0 and Ksat."""
        theta_values = np.linspace(
            sandy_loam_params.theta_r + 0.01,
            sandy_loam_params.theta_s - 0.01,
            20
        )

        for theta in theta_values:
            K = hydraulic_conductivity_from_content(theta, sandy_loam_params)
            assert 0 <= K <= sandy_loam_params.K_sat

    def test_ptf_reasonable_values(self):
        """PTF should produce reasonable parameter values."""
        params = estimate_van_genuchten_params(
            sand_percent=50,
            clay_percent=20,
            organic_matter_percent=2.5,
        )

        assert 0.01 <= params.theta_r <= 0.15
        assert 0.3 <= params.theta_s <= 0.6
        assert 0.001 <= params.alpha <= 1.0
        assert 1.05 <= params.n <= 3.0
        assert 1 <= params.K_sat <= 10000


class TestFeatureEngineeringConversions:
    def test_feature_engineer_uses_vg_params_for_conversions(self):
        vg = VanGenuchtenParams(
            theta_r=0.04,
            theta_s=0.42,
            alpha=0.06,
            n=1.7,
            K_sat=500,
        )

        # Intentionally set conflicting scalar config values; vg_params should win.
        cfg = FeatureConfig(
            theta_sat=0.60,
            theta_res=0.10,
            alpha_vg=0.2,
            n_vg=3.0,
        )

        fe = FeatureEngineer(cfg, vg_params=vg)

        psi = pd.Series([-10.0, -33.0, -100.0])
        theta_expected = pd.Series(
            [water_content_from_potential(p, vg) for p in psi],
            index=psi.index,
        )
        theta_actual = fe._psi_to_theta(psi)

        assert np.allclose(theta_actual.values,
                           theta_expected.values, atol=1e-10)

        # Roundtrip θ -> ψ should match canonical conversion
        theta_in = pd.Series([0.20, 0.25, 0.30])
        psi_expected = pd.Series(
            [potential_from_water_content(t, vg) for t in theta_in],
            index=theta_in.index,
        )
        psi_actual = fe._theta_to_psi(theta_in)
        assert np.allclose(psi_actual.values, psi_expected.values, atol=1e-10)


# =============================================================================
# WATER BALANCE TESTS
# =============================================================================

class TestWaterBalance:
    """Tests for water balance model."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple single-layer model."""
        return create_water_balance_model(
            sand_percent=40,
            clay_percent=30,
            n_layers=1,
            max_depth_m=0.5,
            initial_psi_kpa=-50.0,
        )

    def test_model_initialization(self, simple_model):
        """Model should initialize with correct state."""
        assert len(simple_model.layers) == 1
        assert simple_model.layers[0].psi_kpa < 0  # Negative potential

    def test_precipitation_increases_wetness(self, simple_model):
        """Precipitation should make soil wetter (less negative psi)."""
        initial_psi = simple_model.layers[0].psi_kpa

        output = simple_model.step(
            current_date=date(2024, 6, 15),
            precipitation_mm=20.0,
            et0_mm=3.0,
        )

        # Potential should be closer to zero (wetter)
        assert simple_model.layers[0].psi_kpa > initial_psi

    def test_et_decreases_wetness(self, simple_model):
        """ET should make soil drier (more negative psi)."""
        initial_psi = simple_model.layers[0].psi_kpa

        output = simple_model.step(
            current_date=date(2024, 6, 15),
            precipitation_mm=0.0,
            et0_mm=8.0,  # High ET
        )

        # Potential should be more negative (drier)
        assert simple_model.layers[0].psi_kpa < initial_psi

    def test_water_balance_closure(self, simple_model):
        """Water balance error should be small."""
        output = simple_model.step(
            current_date=date(2024, 6, 15),
            precipitation_mm=10.0,
            et0_mm=5.0,
        )

        # Balance error should be less than 1 mm
        assert abs(output.water_balance_error_mm) < 1.0

    def test_multi_layer_model(self):
        """Multi-layer model should run correctly."""
        model = create_water_balance_model(
            sand_percent=50,
            clay_percent=20,
            n_layers=3,
            max_depth_m=1.0,
        )

        assert len(model.layers) == 3

        # Run a few days
        dates = [date(2024, 6, i) for i in range(1, 11)]
        precip = [0, 0, 15, 5, 0, 0, 0, 20, 0, 0]
        et0 = [5, 5, 3, 2, 5, 6, 6, 3, 5, 5]

        results = model.run_period(dates, precip, et0, warmup_days=0)

        assert len(results) == 10


# =============================================================================
# EVAPOTRANSPIRATION TESTS
# =============================================================================

class TestEvapotranspiration:
    """Tests for ET calculations."""

    def test_crop_coefficients_for_maize(self):
        """Maize crop coefficients should have expected values."""
        coef = CropCoefficients.for_crop("maize")

        assert coef.Kcb_ini < coef.Kcb_mid  # Initial < mid
        assert coef.Kcb_end < coef.Kcb_mid  # End < mid
        assert coef.total_season == coef.L_ini + coef.L_dev + coef.L_mid + coef.L_late

    def test_kcb_from_growth_stage(self):
        """Kcb should vary with growth stage."""
        coef = CropCoefficients.for_crop("maize")
        planting_doy = 100

        # Initial stage
        Kcb_ini = get_Kcb_from_doy(110, planting_doy, coef)

        # Mid-season
        Kcb_mid = get_Kcb_from_doy(
            planting_doy + coef.L_ini + coef.L_dev + 10,
            planting_doy,
            coef
        )

        # Mid should be higher than initial
        assert Kcb_mid > Kcb_ini

    def test_kcb_from_ndvi(self):
        """Kcb should increase with NDVI."""
        Kcb_low = get_Kcb_from_ndvi(0.2)
        Kcb_high = get_Kcb_from_ndvi(0.8)

        assert Kcb_high > Kcb_low

    def test_water_stress_coefficient(self):
        """Ks should decrease when soil is dry."""
        theta_fc = 0.30
        theta_wp = 0.10

        # Wet soil
        Ks_wet = compute_water_stress_coefficient(0.28, theta_fc, theta_wp)
        # Dry soil
        Ks_dry = compute_water_stress_coefficient(0.12, theta_fc, theta_wp)

        assert Ks_wet > Ks_dry
        assert 0 <= Ks_dry <= 1
        assert 0 <= Ks_wet <= 1

    def test_et_partitioning(self):
        """ET should partition into E and T."""
        result = compute_et_partitioning(
            et0_mm=5.0,
            ndvi=0.6,
            theta_surface=0.25,
            theta_root=0.22,
            theta_fc=0.30,
            theta_wp=0.10,
        )

        # Total should be reasonable
        assert 0 < result["et_actual_mm"] < 8.0

        # Sum of parts should equal total
        assert abs(
            result["evaporation_mm"] + result["transpiration_mm"]
            - result["et_actual_mm"]
        ) < 0.01


# =============================================================================
# TROPICAL CORRECTIONS TESTS
# =============================================================================

class TestTropicalCorrections:
    """Tests for tropical soil corrections."""

    @pytest.fixture
    def ferralsol_corrections(self):
        """Corrections for typical ferralsol."""
        return TropicalSoilCorrections.for_african_soil(
            soil_type="ferralsol",
            clay_fraction=0.45,
            sand_fraction=0.30,
            organic_carbon_pct=2.0,
        )

    def test_oxide_aggregation_increases_macroporosity(self, ferralsol_corrections):
        """Higher oxide content should increase aggregation factor."""
        agg_factor = ferralsol_corrections.get_oxide_aggregation_factor()
        assert agg_factor > 1.0  # Should increase porosity

    def test_theta_sat_correction(self, ferralsol_corrections):
        """Corrected theta_s should be higher for ferralsols."""
        original = 0.40
        corrected = ferralsol_corrections.correct_theta_sat(original)

        assert corrected > original
        assert corrected < 0.65  # Physical limit

    def test_ksat_correction(self, ferralsol_corrections):
        """Corrected Ksat should be higher for ferralsols."""
        original = 100.0  # mm/day
        corrected = ferralsol_corrections.correct_Ksat(original)

        assert corrected > original

    def test_infiltration_partitioning(self):
        """Infiltration should partition into matrix and macropore flow."""
        corrections = TropicalSoilCorrections.for_african_soil(
            soil_type="ferralsol",
            clay_fraction=0.40,
            sand_fraction=0.35,
        )

        result = partition_infiltration(
            precip_mm=30.0,
            precip_duration_hr=2.0,
            Ksat_mm_day=200.0,
            theta_current=0.25,
            theta_sat=0.45,
            tropical_corrections=corrections,
        )

        # Total should sum to input (minus runoff)
        assert abs(
            result["matrix_mm"] + result["macropore_mm"] +
            result["runoff_mm"] - 30.0
        ) < 0.01


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for feature engineering."""
        n_hours = 500
        dates = pd.date_range(start="2024-01-01", periods=n_hours, freq="H")

        df = pd.DataFrame({
            "psi_kpa": -50 + np.random.randn(n_hours) * 10,
            "temperature_2m": 25 + np.random.randn(n_hours) * 5,
            "relative_humidity_2m": 60 + np.random.randn(n_hours) * 15,
            "precipitation": np.maximum(0, np.random.randn(n_hours) * 2 - 1),
            "evapotranspiration": np.maximum(0, 0.2 + np.random.randn(n_hours) * 0.1),
        }, index=dates)

        return df

    def test_lag_features_created(self, sample_data):
        """Lag features should be created."""
        engineer = FeatureEngineer()
        result = engineer.create_features(sample_data, psi_col="psi_kpa")

        assert "psi_lag_24h" in result.columns
        assert "psi_lag_72h" in result.columns

    def test_rolling_features_created(self, sample_data):
        """Rolling window features should be created."""
        engineer = FeatureEngineer()
        result = engineer.create_features(sample_data, psi_col="psi_kpa")

        assert "psi_rolling_mean_24h" in result.columns
        assert "temperature_2m_mean_72h" in result.columns

    def test_temporal_features_created(self, sample_data):
        """Temporal features should be created."""
        config = FeatureConfig(include_temporal=True)
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_data, psi_col="psi_kpa")

        assert "hour_sin" in result.columns
        assert "doy_sin" in result.columns

    def test_stress_indices_created(self, sample_data):
        """Stress indices should be created."""
        config = FeatureConfig(include_stress_indices=True)
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_data, psi_col="psi_kpa")

        assert "stress_flag" in result.columns or "relative_soil_moisture" in result.columns


# =============================================================================
# ML MODEL TESTS
# =============================================================================

class TestHybridModel:
    """Tests for hybrid ML model."""

    @pytest.fixture
    def training_data(self):
        """Create synthetic training data."""
        n = 1000
        np.random.seed(42)

        # Features
        X = pd.DataFrame({
            "precip_24h": np.random.exponential(2, n),
            "et_24h": np.random.uniform(2, 6, n),
            "psi_lag_24h": -50 + np.random.randn(n) * 20,
            "temp_mean": 25 + np.random.randn(n) * 5,
        })

        # Physics predictions (simple model)
        y_physics = -50 + 0.5 * X["precip_24h"] - 1.0 * X["et_24h"]

        # True values (physics + some pattern)
        y_observed = y_physics + 5 * \
            np.sin(X["psi_lag_24h"] / 20) + np.random.randn(n) * 3

        return X, y_observed.values, y_physics.values

    def test_model_fitting(self, training_data):
        """Model should fit without errors."""
        X, y_obs, y_phys = training_data

        config = HybridModelConfig(
            n_estimators=100,
            use_ensemble=False,
        )
        model = ResidualLearner(config)
        model.fit(X, y_obs, y_phys)

        assert model.is_fitted

    def test_prediction_shape(self, training_data):
        """Predictions should have correct shape."""
        X, y_obs, y_phys = training_data

        config = HybridModelConfig(n_estimators=100, use_ensemble=False)
        model = ResidualLearner(config)
        model.fit(X, y_obs, y_phys)

        result = model.predict(X[:100], y_phys[:100])

        assert len(result["prediction"]) == 100
        assert "residual" in result

    def test_predictions_improve_over_physics(self, training_data):
        """Hybrid predictions should be better than physics alone."""
        X, y_obs, y_phys = training_data

        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_obs_train, y_obs_test = y_obs[:split], y_obs[split:]
        y_phys_train, y_phys_test = y_phys[:split], y_phys[split:]

        config = HybridModelConfig(n_estimators=200, use_ensemble=False)
        model = ResidualLearner(config)
        model.fit(X_train, y_obs_train, y_phys_train)

        result = model.predict(X_test, y_phys_test)

        # Compute RMSE
        rmse_physics = np.sqrt(np.mean((y_obs_test - y_phys_test) ** 2))
        rmse_hybrid = np.sqrt(
            np.mean((y_obs_test - result["prediction"]) ** 2))

        # Hybrid should be better
        assert rmse_hybrid < rmse_physics


# =============================================================================
# VALIDATION METRICS TESTS
# =============================================================================

class TestValidationMetrics:
    """Tests for validation metrics."""

    def test_kge_perfect_prediction(self):
        """KGE should be 1.0 for perfect predictions."""
        obs = np.array([1, 2, 3, 4, 5])
        pred = np.array([1, 2, 3, 4, 5])

        kge, r, alpha, beta = compute_kge(obs, pred)

        assert abs(kge - 1.0) < 0.001

    def test_nse_perfect_prediction(self):
        """NSE should be 1.0 for perfect predictions."""
        obs = np.array([1, 2, 3, 4, 5])
        pred = np.array([1, 2, 3, 4, 5])

        nse = compute_nse(obs, pred)

        assert abs(nse - 1.0) < 0.001

    def test_nse_mean_prediction(self):
        """NSE should be 0 when predicting mean."""
        obs = np.array([1, 2, 3, 4, 5])
        pred = np.array([3, 3, 3, 3, 3])  # Mean

        nse = compute_nse(obs, pred)

        assert abs(nse) < 0.001


# =============================================================================
# PSI-DRIVEN ET STRESS TESTS
# =============================================================================

class TestPsiDrivenETStress:
    """Tests for ψ-driven water stress coefficient."""

    def test_no_stress_above_field_capacity(self):
        """Ks should be 1.0 at field capacity."""
        from swpps.physics.evapotranspiration import compute_water_stress_from_potential

        Ks = compute_water_stress_from_potential(psi_kpa=-10.0)
        assert abs(Ks - 1.0) < 0.01

    def test_zero_stress_at_wilting_point(self):
        """Ks should be 0 at wilting point."""
        from swpps.physics.evapotranspiration import compute_water_stress_from_potential

        Ks = compute_water_stress_from_potential(psi_kpa=-1500.0)
        assert Ks < 0.01

    def test_stress_increases_with_drying(self):
        """Ks should decrease as soil dries (ψ becomes more negative)."""
        from swpps.physics.evapotranspiration import compute_water_stress_from_potential

        Ks_wet = compute_water_stress_from_potential(psi_kpa=-50.0)
        Ks_med = compute_water_stress_from_potential(psi_kpa=-200.0)
        Ks_dry = compute_water_stress_from_potential(psi_kpa=-800.0)

        assert Ks_wet > Ks_med > Ks_dry


# =============================================================================
# RESIDUAL DIAGNOSTICS TESTS
# =============================================================================

class TestResidualDiagnostics:
    """Tests for residual-based calibration diagnostics."""

    def test_detect_dry_bias(self):
        """Should detect when physics is too dry."""
        from swpps.calibration.calibrate import ResidualAnalyzer

        # Physics predicting too dry (ψ too negative)
        psi_obs = np.array([-30, -40, -50, -60, -70])  # Observed
        psi_phys = np.array([-50, -60, -70, -80, -90])  # Physics too negative

        analyzer = ResidualAnalyzer()
        diag = analyzer.analyze(psi_obs, psi_phys)

        # Mean residual should be positive (physics too dry)
        assert diag.mean_residual > 10
        assert diag.dry_bias_fraction > 0.5

    def test_detect_wet_bias(self):
        """Should detect when physics is too wet."""
        from swpps.calibration.calibrate import ResidualAnalyzer

        # Physics predicting too wet (ψ not negative enough)
        psi_obs = np.array([-80, -90, -100, -110, -120])  # Observed
        psi_phys = np.array([-50, -60, -70, -80, -90])    # Physics too wet

        analyzer = ResidualAnalyzer()
        diag = analyzer.analyze(psi_obs, psi_phys)

        # Mean residual should be negative (physics too wet)
        assert diag.mean_residual < -10
        assert diag.wet_bias_fraction > 0.5

    def test_suggests_ksat_increase_when_infiltration_low(self):
        """Should suggest Ksat increase when physics is too dry during rain."""
        from swpps.calibration.calibrate import ResidualAnalyzer

        n = 100
        psi_obs = np.full(n, -40.0)
        psi_phys = np.full(n, -70.0)  # Too dry
        precip = np.where(np.arange(n) < 30, 10.0, 0.0)  # Rain at start

        analyzer = ResidualAnalyzer()
        diag = analyzer.analyze(psi_obs, psi_phys, precipitation=precip)

        assert diag.ksat_adjustment == "increase"


# =============================================================================
# THETA OUTPUT CONVERSION TESTS
# =============================================================================

class TestThetaOutputConversion:
    """Tests for ψ → θ output conversion."""

    def test_physics_output_theta_conversion(self):
        """PhysicsModelOutput should convert ψ to θ."""
        from swpps.core.types import PhysicsModelOutput
        from swpps.physics.van_genuchten import VanGenuchtenParams as VGParams

        output = PhysicsModelOutput(
            date=date(2024, 6, 15),
            psi_surface_kpa=-50.0,
            psi_root_kpa=-80.0,
        )

        vg = VGParams(
            theta_r=0.05, theta_s=0.40, alpha=0.05, n=1.5, K_sat=100.0
        )

        output.compute_theta_from_psi(vg)

        assert output.theta_surface is not None
        assert output.theta_root is not None
        assert 0.05 < output.theta_surface < 0.40
        assert 0.05 < output.theta_root < 0.40
        # Root should be drier (lower θ for more negative ψ)
        assert output.theta_root < output.theta_surface

    def test_prediction_result_theta_conversion(self):
        """PredictionResult should convert ψ to θ."""
        from swpps.core.types import PredictionResult, SoilMoistureStatus

        result = PredictionResult(
            timestamp=datetime.now(),
            horizon_hours=24,
            psi_predicted_kpa=-75.0,
            psi_lower_bound_kpa=-120.0,
            psi_upper_bound_kpa=-40.0,
            psi_physics_kpa=-70.0,
            psi_ml_residual_kpa=-5.0,
            status=SoilMoistureStatus.OPTIMAL,
            confidence=0.85,
            uncertainty_kpa=15.0,
        )

        result.compute_theta()

        assert result.theta_predicted is not None
        assert 0.0 < result.theta_predicted < 0.5


# =============================================================================
# PHYSICS-INFORMED FEATURES TESTS
# =============================================================================

class TestPhysicsInformedFeatures:
    """Tests for physics-informed ML features."""

    def test_infiltration_features_created(self):
        """Should create infiltration physics features."""
        config = FeatureConfig(include_physics=True)
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            "psi_kpa": np.linspace(-30, -100, 100),
            "precipitation": np.random.exponential(2, 100),
            "temperature_2m": np.random.normal(25, 5, 100),
        }, index=pd.date_range("2024-01-01", periods=100, freq="h"))

        features = engineer.create_features(df, psi_col="psi_kpa")

        # Check infiltration features exist
        assert "infiltration_capacity_index" in features.columns
        assert "infiltration_efficiency" in features.columns

    def test_drainage_features_created(self):
        """Should create drainage physics features."""
        config = FeatureConfig(include_physics=True)
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            "psi_kpa": np.linspace(-10, -100, 100),
            "precipitation": np.random.exponential(2, 100),
            "temperature_2m": np.random.normal(25, 5, 100),
        }, index=pd.date_range("2024-01-01", periods=100, freq="h"))

        features = engineer.create_features(df, psi_col="psi_kpa")

        # Check drainage features exist
        assert "above_fc" in features.columns
        assert "drainage_potential" in features.columns
        assert "rel_conductivity_proxy" in features.columns

    def test_psi_physics_state_features(self):
        """Should create ψ_phys, θ_phys, and related features."""
        config = FeatureConfig(include_physics_informed=True)
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            "psi_kpa": np.linspace(-30, -200, 200),
            "psi_physics_kpa": np.linspace(-35, -210, 200),
            "precipitation": np.random.exponential(2, 200),
            "evapotranspiration": np.random.uniform(0.1, 0.5, 200),
            "temperature_2m": np.random.normal(25, 5, 200),
        }, index=pd.date_range("2024-01-01", periods=200, freq="h"))

        features = engineer.create_features(
            df, psi_col="psi_kpa",
            physics_cols=["psi_physics_kpa"]
        )

        # Check core physics features
        assert "psi_obs" in features.columns
        assert "theta_obs" in features.columns
        assert "dpsi_dtheta" in features.columns
        assert "psi_phys" in features.columns
        assert "theta_phys" in features.columns

    def test_hydraulic_conductivity_features(self):
        """Should create K(θ), Se features."""
        config = FeatureConfig(
            include_physics_informed=True,
            Ksat_mm_day=200.0,
            theta_sat=0.45,
            theta_res=0.05,
        )
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            "psi_kpa": np.linspace(-10, -500, 100),
            "precipitation": np.zeros(100),
            "temperature_2m": np.full(100, 25.0),
        }, index=pd.date_range("2024-01-01", periods=100, freq="h"))

        features = engineer.create_features(df, psi_col="psi_kpa")

        # Check K features
        assert "Se_obs" in features.columns
        assert "K_theta_obs" in features.columns
        assert "log_K_theta_obs" in features.columns
        assert "Ksat_mm_day" in features.columns

        # K should decrease as soil dries
        K_wet = features["K_theta_obs"].iloc[0]  # ψ = -10 kPa
        K_dry = features["K_theta_obs"].iloc[-1]  # ψ = -500 kPa
        assert K_wet > K_dry

    def test_et_stress_features(self):
        """Should create f(ψ) stress function features."""
        config = FeatureConfig(
            include_physics_informed=True,
            stress_psi_kpa=-100.0,
            wilting_psi_kpa=-1500.0,
        )
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            "psi_kpa": np.array([-20, -50, -100, -500, -1000, -1500]),
            "evapotranspiration": np.full(6, 0.3),
            "temperature_2m": np.full(6, 25.0),
        }, index=pd.date_range("2024-01-01", periods=6, freq="h"))

        features = engineer.create_features(df, psi_col="psi_kpa")

        # Check stress features
        assert "f_psi_stress" in features.columns
        assert "ET_actual" in features.columns

        # Stress should increase (f decreases) as soil dries
        f_wet = features["f_psi_stress"].iloc[0]  # ψ = -20 (no stress)
        f_dry = features["f_psi_stress"].iloc[-1]  # ψ = -1500 (full stress)

        assert f_wet > 0.9  # No stress above critical
        assert f_dry < 0.1  # Full stress at wilting

    def test_mass_balance_diagnostics(self):
        """Should create water balance diagnostic features."""
        config = FeatureConfig(include_physics_informed=True)
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            "psi_kpa": np.linspace(-30, -80, 200),
            "psi_physics_kpa": np.linspace(-35, -90, 200),
            "precipitation": np.where(np.arange(200) < 24, 2.0, 0.0),
            "evapotranspiration": np.full(200, 0.2),
            "temperature_2m": np.full(200, 25.0),
        }, index=pd.date_range("2024-01-01", periods=200, freq="h"))

        features = engineer.create_features(
            df, psi_col="psi_kpa",
            physics_cols=["psi_physics_kpa"]
        )

        # Check mass balance features
        assert "cum_rain_24h" in features.columns
        assert "water_balance_24h" in features.columns
        assert "storage_change_discrepancy" in features.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

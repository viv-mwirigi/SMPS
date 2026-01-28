"""
Physics modules for soil moisture prediction.

PRODUCTION MODEL (Recommended for farmers):
==========================================
Use `create_water_balance_model()` - this creates the SimpleWaterBalance model
with physics-based soil moisture predictions.

    >>> from smps.physics import create_water_balance_model
    >>> model = create_water_balance_model(crop_type="maize")
    >>> fluxes, output_theta = model.run_daily(precipitation_mm=15, et0_mm=5, ndvi=0.6)

The production model includes:
  1. Van Genuchten water retention curves
  2. Darcy-based drainage
  3. FAO-56 evapotranspiration partitioning
  4. Root zone water uptake
  5. Multi-layer soil profile simulation

Legacy Models (for testing/comparison only):
- SimpleWaterBalance: Current production model with configurable complexity

Supporting Modules:
- soil_hydraulics: Van Genuchten, hydraulic functions
- pedotransfer: PTFs with tropical corrections
- adaptive_calibration: Site-specific parameter adjustments

Note: The SimpleWaterBalance model provides the recommended physics capabilities.
Use create_water_balance_model() for the production model.
"""
from smps.physics.pedotransfer import (
    TextureClass,
    classify_soil_texture,
    # Gap 6: Tropical soil corrections
    TropicalSoilCorrections,
    SoilParameterDistribution,
    estimate_soil_parameters_tropical,
    create_parameter_distribution,
)

# Enhanced physics modules (v2)
from smps.physics.soil_hydraulics import (
    VanGenuchtenParameters,
    BrooksCoreyParameters,
    FeddesParameters,
    van_genuchten_theta_from_psi,
    van_genuchten_psi_from_theta,
    van_genuchten_mualem_K,
    feddes_stress_factor,
    theta_at_field_capacity,
    theta_at_wilting_point,
    plant_available_water,
)
from smps.physics.infiltration import (
    GreenAmptParameters,
    InfiltrationState,
    RainfallIntensityDistribution,
    daily_infiltration_green_ampt,
    green_ampt_infiltration_rate,
    time_to_ponding,
)
from smps.physics.evapotranspiration import (
    CropCoefficientCurve,
    SoilEvaporationState,
    ETResult,
    InterceptionParameters,
    calculate_et_fao56_dual,
    ndvi_to_lai,
    calculate_Ks,
)
from smps.physics.root_uptake import (
    RootDistributionParameters,
    RootWaterUptakeModel,
    RootUptakeResult,
)
from smps.physics.vertical_flux import (
    VerticalFluxParameters,
    VerticalFluxModel,
    VerticalFluxResult,
    LayerState,
)

# Gap 7: Dynamic crop development
from smps.physics.crop_development import (
    GrowthStage,
    PhenologyParameters,
    RootGrowthParameters,
    ResidueCoverParameters,
    CropState,
    CropDevelopmentModel,
    create_crop_model,
    estimate_planting_window,
)

# Gap 8 & 9: Numerical solver improvements
from smps.physics.numerical_solver import (
    TimestepController,
    TimestepMode,
    MassBalanceState,
    ImplicitEulerSolver,
    AdaptiveWaterBalanceSolver,
    create_adaptive_solver,
    validate_mass_balance,
)

# Adaptive calibration and improved physics (v2.4)
from smps.physics.adaptive_calibration import (
    AdaptivePhysicsCalibrator,
    AdaptiveCalibrationParameters,
    SiteCharacteristics,
    ClimateZone,
    LandCoverType,
    create_site_calibrator,
    tropical_ptf_van_genuchten,
)

# Simple water balance model (current production model)
from smps.physics.simple_water_balance import (
    SimpleWaterBalance,
    ModelConfig,
    LayerConfig,
    SoilHydraulicParams,
    create_simple_config_improved,
    create_simple_config,
)

__all__ = [
    # Pedotransfer (v2.1 - Gap 6)
    "TextureClass",
    "classify_soil_texture",
    "TropicalSoilCorrections",
    "SoilParameterDistribution",
    "estimate_soil_parameters_tropical",
    "create_parameter_distribution",

    # Soil hydraulics
    "VanGenuchtenParameters",
    "BrooksCoreyParameters",
    "FeddesParameters",
    "van_genuchten_theta_from_psi",
    "van_genuchten_psi_from_theta",
    "van_genuchten_mualem_K",
    "feddes_stress_factor",
    "theta_at_field_capacity",
    "theta_at_wilting_point",
    "plant_available_water",

    # Infiltration
    "GreenAmptParameters",
    "InfiltrationState",
    "RainfallIntensityDistribution",
    "daily_infiltration_green_ampt",
    "green_ampt_infiltration_rate",
    "time_to_ponding",

    # Evapotranspiration
    "CropCoefficientCurve",
    "SoilEvaporationState",
    "ETResult",
    "calculate_et_fao56_dual",
    "ndvi_to_lai",
    "calculate_Ks",

    # Root uptake
    "RootDistributionParameters",
    "RootWaterUptakeModel",
    "RootUptakeResult",

    # Vertical flux
    "VerticalFluxParameters",
    "VerticalFluxModel",
    "VerticalFluxResult",
    "LayerState",

    # Crop development (v2.1 - Gap 7)
    "GrowthStage",
    "PhenologyParameters",
    "RootGrowthParameters",
    "ResidueCoverParameters",
    "CropState",
    "CropDevelopmentModel",
    "create_crop_model",
    "estimate_planting_window",

    # Numerical solver (v2.1 - Gap 8 & 9)
    "TimestepController",
    "TimestepMode",
    "MassBalanceState",
    "ImplicitEulerSolver",
    "AdaptiveWaterBalanceSolver",
    "create_adaptive_solver",
    "validate_mass_balance",

    # Simple water balance model (current production)
    "SimpleWaterBalance",
    "ModelConfig",
    "LayerConfig",
    "SoilHydraulicParams",
    "create_simple_config_improved",
    "create_simple_config",

    # Main factory function (RECOMMENDED)
    "create_water_balance_model",
]


# =============================================================================
# UNIFIED FACTORY FUNCTION - USE THIS FOR PRODUCTION
# =============================================================================

def create_water_balance_model(
    crop_type: str = "maize",
    n_layers: int = 5,
    soil_texture: str = "loam",
    use_full_physics: bool = True,
    soil_params=None,
    soil_param_method: str = "texture_class",
    theta_s_adj: float = 1.0,
    Ks_adj: float = 1.0,
    alpha_adj: float = 1.0,
    **kwargs
):
    """
    Create a water balance model for soil moisture prediction.

    This creates the SimpleWaterBalance model which provides physics-based
    soil moisture predictions with configurable complexity.

    Args:
        crop_type: Crop type for root parameters ("maize", "wheat", etc.) - currently not used
        n_layers: Number of soil layers (default: 3 for 0-100cm)
        soil_texture: Soil texture class ("sand", "loam", "clay", etc.)
        use_full_physics: If True (default), use full physics. If False, use simpler settings.
        soil_params: Optional soil parameters dict with sand_pct, clay_pct, etc.
        **kwargs: Additional parameters passed to model

    Returns:
        SimpleWaterBalance model instance

    Example:
        >>> model = create_water_balance_model(crop_type="maize")
        >>> fluxes, output_theta = model.run_daily(
        ...     precipitation_mm=15.0,
        ...     et0_mm=5.0,
        ...     ndvi=0.6,
        ...     irrigation_mm=5.0  # NEW: Irrigation water input
        ... )
        >>> print(f"Soil moisture: {output_theta:.3f} m³/m³")
    """
    # Extract soil parameters
    if soil_params is not None:
        sand_percent = soil_params.get(
            'sand_pct', soil_params.get('sand_percent', 40))
        clay_percent = soil_params.get(
            'clay_pct', soil_params.get('clay_percent', 20))
    else:
        # Default soil parameters based on texture
        texture_defaults = {
            'sand': (90, 5),
            'loamy_sand': (80, 10),
            'sandy_loam': (65, 15),
            'loam': (40, 20),
            'silt_loam': (20, 25),
            'sandy_clay_loam': (55, 30),
            'clay_loam': (35, 35),
            'silty_clay_loam': (15, 40),
            'sandy_clay': (50, 45),
            'silty_clay': (10, 50),
            'clay': (20, 60)
        }
        sand_percent, clay_percent = texture_defaults.get(
            soil_texture.lower(), (40, 20))

    # Create model config
    config = create_simple_config_improved(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        output_depth_m=0.10,
        # SimpleWaterBalance works best with fewer layers
        n_layers=min(n_layers, 5),
        max_depth_m=1.0,
        vegetation_fraction=kwargs.get('vegetation_fraction', 0.5),
        latitude=kwargs.get('latitude'),
        longitude=kwargs.get('longitude'),
        use_tropical_ptf=True,
        apply_adaptive_calibration=use_full_physics,
    )

    return SimpleWaterBalance(config)

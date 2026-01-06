"""
Physics modules for soil moisture prediction.

This package contains:

Original Models (v1):
- TwoBucketWaterBalance: Simple two-layer model
- ThreeLayerWaterBalance: Extended 0-100cm model

Enhanced Physics (v2):
- EnhancedWaterBalance: Physics-improved model addressing 5 critical gaps:
  1. Green-Ampt infiltration with rainfall intensity
  2. FAO-56 dual crop coefficient for ET partitioning
  3. Feddes pressure head-based root water uptake
  4. Darcy percolation with Van Genuchten-Mualem K(θ)
  5. Gradient-driven capillary rise

Parameterization Improvements (v2.1):
- Gap 6: Tropical soil corrections (OM, structure factor)
- Gap 7: Dynamic crop development with GDD-based parameters

Supporting Modules:
- soil_hydraulics: Van Genuchten, Feddes, hydraulic functions
- infiltration: Green-Ampt model
- evapotranspiration: FAO-56 dual Kc model
- root_uptake: Feddes root water uptake
- vertical_flux: Darcy-based percolation and capillary rise
- crop_development: GDD-based phenology and root growth
- pedotransfer: PTFs with tropical corrections
"""
from smps.physics.water_balance import (
    TwoBucketWaterBalance,
    SiteSpecificWaterBalance,
    ModelParameters,
    BucketState,
    Fluxes,
    # Three-layer model for 0-100cm depth (FLDAS compatible)
    ThreeLayerWaterBalance,
    ThreeLayerParameters,
    create_three_layer_model,
)
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
from smps.physics.enhanced_water_balance import (
    EnhancedWaterBalance,
    EnhancedModelParameters,
    EnhancedLayerState,
    EnhancedFluxes,
    create_enhanced_model,
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

__all__ = [
    # Original models (v1)
    "TwoBucketWaterBalance",
    "SiteSpecificWaterBalance",
    "ModelParameters",
    "BucketState",
    "Fluxes",
    "ThreeLayerWaterBalance",
    "ThreeLayerParameters",
    "create_three_layer_model",

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

    # Enhanced model (v2)
    "EnhancedWaterBalance",
    "EnhancedModelParameters",
    "EnhancedLayerState",
    "EnhancedFluxes",
    "create_enhanced_model",

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
]

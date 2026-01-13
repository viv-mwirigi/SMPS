"""
Physics modules for soil moisture prediction.

PRODUCTION MODEL (Recommended for farmers):
==========================================
Use `create_water_balance_model()` - this creates the full-physics
EnhancedWaterBalance model optimized for accurate predictions.

    >>> from smps.physics import create_water_balance_model
    >>> model = create_water_balance_model(crop_type="maize")
    >>> result, fluxes = model.run_daily(precipitation_mm=15, et0_mm=5, ndvi=0.6)

The production model includes:
  1. Green-Ampt infiltration - accurate runoff during storms
  2. FAO-56 dual Kc - proper ET partitioning by growth stage
  3. Feddes root uptake - realistic drought stress timing
  4. Darcy vertical flux - correct soil water redistribution
  5. Capillary rise - dry season moisture from water table

Legacy Models (for testing/comparison only):
- TwoBucketWaterBalance: Simple 2-layer, fast but less accurate
- ThreeLayerWaterBalance: Simple 3-layer, used in initial ISMN validation

Supporting Modules:
- soil_hydraulics: Van Genuchten, Feddes, hydraulic functions
- infiltration: Green-Ampt model
- evapotranspiration: FAO-56 dual Kc model
- root_uptake: Feddes root water uptake
- vertical_flux: Darcy-based percolation and capillary rise
- crop_development: GDD-based phenology and root growth
- pedotransfer: PTFs with tropical corrections

Note: The legacy TwoBucketWaterBalance and ThreeLayerWaterBalance have been
replaced by EnhancedWaterBalance which provides full physics capabilities.
Use create_water_balance_model() for the recommended production model.
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

    This is the RECOMMENDED way to create a model for farmer-facing
    prediction systems. By default, it creates the full-physics model
    for maximum accuracy.

    Args:
        crop_type: Crop type for root parameters ("maize", "wheat", etc.)
        n_layers: Number of soil layers (default: 3 for 0-100cm)
        soil_texture: Soil texture class ("sand", "loam", "clay", etc.)
        use_full_physics: If True (default), use EnhancedWaterBalance.
                         If False, use simpler ThreeLayerWaterBalance.
        **kwargs: Additional parameters passed to model

    Returns:
        Water balance model instance

    Example:
        >>> model = create_water_balance_model(crop_type="maize")
        >>> result, fluxes = model.run_daily(
        ...     precipitation_mm=15.0,
        ...     et0_mm=5.0,
        ...     ndvi=0.6
        ... )
        >>> print(f"Soil moisture: {result.theta_surface:.3f} m³/m³")
    """
    if use_full_physics:
        # Production model with full physics
        from smps.physics.soil_hydraulics import VanGenuchtenParameters

        # Honor explicit user-provided VG parameters.
        explicit_vg_params = kwargs.pop("vg_params", None)

        if soil_params is not None:
            params = EnhancedModelParameters.from_soil_parameters(
                soil_params,
                crop_type=crop_type,
                **kwargs,
            )

        elif explicit_vg_params is not None:
            params = EnhancedModelParameters(
                n_layers=n_layers,
                crop_type=crop_type,
                vg_params=list(explicit_vg_params),
                use_green_ampt=True,
                use_fao56_dual=True,
                use_feddes_uptake=True,
                use_darcy_flux=True,
                enable_capillary_rise=True,
                **kwargs,
            )

        else:
            # If requested, build VG params using pedotransfer functions.
            if soil_param_method.lower() in {"saxton", "rosetta", "tropical"}:
                from smps.physics.pedotransfer import (
                    estimate_soil_parameters_saxton,
                    estimate_soil_parameters_rosetta,
                    estimate_soil_parameters_tropical,
                )

                sand_percent = kwargs.pop("sand_percent", None)
                clay_percent = kwargs.pop("clay_percent", None)
                organic_matter_percent = kwargs.pop(
                    "organic_matter_percent", 2.0)

                if sand_percent is None or clay_percent is None:
                    raise ValueError(
                        "sand_percent and clay_percent are required when soil_param_method is saxton/rosetta/tropical"
                    )

                method = soil_param_method.lower()
                if method == "saxton":
                    sp = estimate_soil_parameters_saxton(
                        sand_percent=sand_percent,
                        clay_percent=clay_percent,
                        organic_matter_percent=organic_matter_percent,
                    )
                elif method == "rosetta":
                    sp = estimate_soil_parameters_rosetta(
                        sand_percent=sand_percent,
                        clay_percent=clay_percent,
                        organic_matter_percent=organic_matter_percent,
                    )
                else:
                    sp = estimate_soil_parameters_tropical(
                        sand_percent=sand_percent,
                        clay_percent=clay_percent,
                        organic_matter_percent=organic_matter_percent,
                    )

                params = EnhancedModelParameters.from_soil_parameters(
                    sp,
                    crop_type=crop_type,
                    **kwargs,
                )
            else:
                vg_params = [
                    VanGenuchtenParameters.from_texture_class(soil_texture)
                    for _ in range(n_layers)
                ]

                params = EnhancedModelParameters(
                    n_layers=n_layers,
                    crop_type=crop_type,
                    vg_params=vg_params,
                    use_green_ampt=True,
                    use_fao56_dual=True,
                    use_feddes_uptake=True,
                    use_darcy_flux=True,
                    enable_capillary_rise=True,
                    **kwargs
                )

        # Optional per-profile calibration multipliers
        if (theta_s_adj != 1.0) or (Ks_adj != 1.0) or (alpha_adj != 1.0):
            params.vg_params = [
                vg.with_multipliers(
                    theta_s_adj=theta_s_adj,
                    Ks_adj=Ks_adj,
                    alpha_adj=alpha_adj,
                )
                for vg in params.vg_params
            ]

        return EnhancedWaterBalance(params)
    else:
        # Simplified model uses same EnhancedWaterBalance but with simpler settings
        from smps.physics.soil_hydraulics import VanGenuchtenParameters

        explicit_vg_params = kwargs.pop("vg_params", None)

        vg_params = list(explicit_vg_params) if explicit_vg_params is not None else [
            VanGenuchtenParameters.from_texture_class(soil_texture)
            for _ in range(n_layers)
        ]

        if (theta_s_adj != 1.0) or (Ks_adj != 1.0) or (alpha_adj != 1.0):
            vg_params = [
                vg.with_multipliers(
                    theta_s_adj=theta_s_adj,
                    Ks_adj=Ks_adj,
                    alpha_adj=alpha_adj,
                )
                for vg in vg_params
            ]

        params = EnhancedModelParameters(
            n_layers=n_layers,
            crop_type=crop_type,
            vg_params=vg_params,
            use_green_ampt=False,  # Simpler infiltration
            use_fao56_dual=False,  # Simpler ET
            use_feddes_uptake=False,  # No stress function
            use_darcy_flux=False,  # Simple drainage
            enable_capillary_rise=False,
            **kwargs
        )
        return EnhancedWaterBalance(params)

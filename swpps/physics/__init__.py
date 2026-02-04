"""Physics module for SWPPS."""

from swpps.physics.van_genuchten import (
    VanGenuchtenParams,
    water_content_from_potential,
    potential_from_water_content,
    hydraulic_conductivity_from_content,
    hydraulic_conductivity_from_potential,
    estimate_van_genuchten_params,
    apply_tropical_corrections,
    tropical_ptf_van_genuchten,
)
from swpps.physics.water_balance import (
    TensionSpaceWaterBalance,
    WaterBalanceConfig,
    LayerConfig,
    LayerState,
    create_water_balance_model,
)
from swpps.physics.evapotranspiration import (
    CropCoefficients,
    get_Kcb_from_doy,
    get_Kcb_from_ndvi,
    ndvi_to_fractional_cover,
    compute_water_stress_coefficient,
    compute_water_stress_from_potential,  # NEW: ψ-driven stress
    compute_soil_evaporation_coefficient,
    compute_et_partitioning,
)
from swpps.physics.tropical import (
    TropicalSoilCorrections,
    estimate_macropore_flow_fraction,
    partition_infiltration,
)

__all__ = [
    # Van Genuchten
    "VanGenuchtenParams",
    "water_content_from_potential",
    "potential_from_water_content",
    "hydraulic_conductivity_from_content",
    "hydraulic_conductivity_from_potential",
    "estimate_van_genuchten_params",
    "apply_tropical_corrections",
    "tropical_ptf_van_genuchten",
    # Water Balance
    "TensionSpaceWaterBalance",
    "WaterBalanceConfig",
    "LayerConfig",
    "LayerState",
    "create_water_balance_model",
    # Evapotranspiration
    "CropCoefficients",
    "get_Kcb_from_doy",
    "get_Kcb_from_ndvi",
    "ndvi_to_fractional_cover",
    "compute_water_stress_coefficient",
    "compute_water_stress_from_potential",  # ψ-driven stress
    "compute_soil_evaporation_coefficient",
    "compute_et_partitioning",
    # Tropical Corrections
    "TropicalSoilCorrections",
    "estimate_macropore_flow_fraction",
    "partition_infiltration",
]

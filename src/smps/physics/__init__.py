"""Physics module for SWPPS."""

from smps.physics.van_genuchten import (
    VanGenuchtenParams,
    water_content_from_potential,
    potential_from_water_content,
    hydraulic_conductivity_from_content,
    hydraulic_conductivity_from_potential,
    estimate_van_genuchten_params,
    apply_tropical_corrections,
    tropical_ptf_van_genuchten,
)
from smps.physics.ivg import (
    IVGParams,
    theta_from_psi as theta_from_psi_ivg,
    psi_from_theta as psi_from_theta_ivg,
    hydraulic_conductivity_from_psi as hydraulic_conductivity_from_psi_ivg,
    ivg_from_vg,
)
from smps.physics.thermal import (
    ThermalParams,
    thermal_conductivity_johansen,
)
from smps.physics.water_balance import (
    TensionSpaceWaterBalance,
    WaterBalanceConfig,
    LayerConfig,
    LayerState,
    create_water_balance_model,
)
from smps.physics.evapotranspiration import (
    CropCoefficients,
    get_Kcb_from_doy,
    get_Kcb_from_ndvi,
    ndvi_to_fractional_cover,
    compute_water_stress_coefficient,
    compute_water_stress_from_potential,  # NEW: ψ-driven stress
    compute_soil_evaporation_coefficient,
    compute_et_partitioning,
)
from smps.physics.tropical import (
    TropicalSoilCorrections,
    estimate_macropore_flow_fraction,
    partition_infiltration,
)
from smps.physics.model import (
    PhysicsModel,
    PhysicsConfig,
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
    # iVG (PDI)
    "IVGParams",
    "theta_from_psi_ivg",
    "psi_from_theta_ivg",
    "hydraulic_conductivity_from_psi_ivg",
    "ivg_from_vg",
    # Thermal
    "ThermalParams",
    "thermal_conductivity_johansen",
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
    # Physics Model
    "PhysicsModel",
    "PhysicsConfig",
]

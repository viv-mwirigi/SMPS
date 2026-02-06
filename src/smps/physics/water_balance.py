"""Tension-space water balance model with iVG + thermal bridge.

Key characteristics:
- Hourly explicit update with Darcy fluxes between layers
- iVG (PDI-style) retention for improved dry-end behavior
- NDVI-driven crop coefficient for dynamic ET
- Johansen thermal conductivity from hydraulic state
"""

import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from smps.core.types import (
    VanGenuchtenParams,
    PhysicsModelOutput,
)
from smps.physics.van_genuchten import (
    estimate_van_genuchten_params,
)
from smps.physics.ivg import (
    IVGParams,
    theta_from_psi,
    psi_from_theta,
    hydraulic_conductivity_from_psi,
    ivg_from_vg,
)
from smps.physics.thermal import (
    ThermalParams,
    thermal_conductivity_johansen,
)


@dataclass
class LayerConfig:
    """Configuration for a single soil layer."""
    depth_top_m: float
    depth_bottom_m: float
    van_genuchten: VanGenuchtenParams
    ivg_params: Optional[IVGParams] = None
    bulk_density_g_cm3: float = 1.35
    quartz_fraction: float = 0.6

    def __post_init__(self):
        if self.ivg_params is None:
            self.ivg_params = ivg_from_vg(
                theta_r=self.van_genuchten.theta_r,
                theta_s=self.van_genuchten.theta_s,
                alpha=self.van_genuchten.alpha,
                n=self.van_genuchten.n,
                K_sat=self.van_genuchten.K_sat,
            )

    @property
    def thickness_m(self) -> float:
        return self.depth_bottom_m - self.depth_top_m

    @property
    def thickness_mm(self) -> float:
        return self.thickness_m * 1000


@dataclass
class LayerState:
    """Current state of a soil layer."""
    config: LayerConfig
    theta: float  # Current volumetric water content
    lambda_phys: float = 0.0

    @property
    def psi_kpa(self) -> float:
        """Current matric potential."""
        return psi_from_theta(self.theta, self.config.ivg_params)

    @property
    def storage_mm(self) -> float:
        """Water storage in mm."""
        return self.theta * self.config.thickness_mm

    @property
    def available_water_mm(self) -> float:
        """Plant available water above wilting point."""
        wp = self.config.van_genuchten.theta_wp
        return max(0, (self.theta - wp) * self.config.thickness_mm)

    @property
    def deficit_to_fc_mm(self) -> float:
        """Water deficit to reach field capacity."""
        fc = self.config.van_genuchten.theta_fc
        return max(0, (fc - self.theta) * self.config.thickness_mm)

    def add_water(self, amount_mm: float) -> float:
        """Add water to layer, return excess."""
        max_add = (self.config.van_genuchten.theta_s -
                   self.theta) * self.config.thickness_mm
        actual_add = min(amount_mm, max_add)
        self.theta += actual_add / self.config.thickness_mm
        return amount_mm - actual_add

    def remove_water(self, amount_mm: float, min_theta: Optional[float] = None) -> float:
        """Remove water from layer, return actual removed."""
        if min_theta is None:
            min_theta = self.config.van_genuchten.theta_r
        max_remove = (self.theta - min_theta) * self.config.thickness_mm
        actual_remove = min(amount_mm, max(0, max_remove))
        self.theta -= actual_remove / self.config.thickness_mm
        return actual_remove


@dataclass
class WaterBalanceConfig:
    """Configuration for water balance model."""
    # Layer configurations
    layers: List[LayerConfig]

    # Root distribution (fraction in each layer)
    root_fractions: Optional[List[float]] = None

    # Time step control
    dt_hours: float = 1.0

    # Infiltration capacity (scales Ksat)
    infiltration_factor: float = 1.0

    # Root water uptake (Feddes-type) thresholds
    psi_anaerobic_kpa: float = -5.0
    psi_optimal_kpa: float = -33.0
    psi_wilting_kpa: float = -1500.0

    # Crop coefficient (NDVI-driven)
    default_crop_coefficient: float = 0.8

    # Darcy flux averaging for inter-layer conductivity
    use_geometric_mean: bool = True

    # Initial conditions
    initial_psi_kpa: float = -50.0  # Initial matric potential

    def __post_init__(self):
        # Default root distribution if not provided
        if self.root_fractions is None:
            n_layers = len(self.layers)
            if n_layers == 1:
                self.root_fractions = [1.0]
            elif n_layers == 2:
                self.root_fractions = [0.6, 0.4]
            elif n_layers == 3:
                self.root_fractions = [0.5, 0.35, 0.15]
            else:
                # Exponential decrease with depth
                self.root_fractions = [0.5 ** i for i in range(n_layers)]
                total = sum(self.root_fractions)
                self.root_fractions = [r / total for r in self.root_fractions]


class TensionSpaceWaterBalance:
    """
    Water balance model that operates in tension space with iVG hydraulics.

    This implementation uses an explicit hourly scheme with Darcy fluxes,
    Feddes-type root uptake, and a Johansen thermal conductivity bridge.
    """

    def __init__(self, config: WaterBalanceConfig):
        self.config = config
        self.layers: List[LayerState] = []
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize layer states from config."""
        self.layers = []
        for layer_config in self.config.layers:
            theta = theta_from_psi(
                self.config.initial_psi_kpa, layer_config.ivg_params)
            theta = max(theta, layer_config.ivg_params.theta_r)
            tparams = ThermalParams(
                bulk_density_g_cm3=layer_config.bulk_density_g_cm3,
                theta_s=layer_config.ivg_params.theta_s,
                theta_r=layer_config.ivg_params.theta_r,
                quartz_fraction=layer_config.quartz_fraction,
            )
            lambda_phys = thermal_conductivity_johansen(theta, tparams)
            self.layers.append(
                LayerState(config=layer_config, theta=theta,
                           lambda_phys=lambda_phys)
            )

    def reset(self, initial_psi_kpa: Optional[float] = None):
        """Reset model state."""
        if initial_psi_kpa is not None:
            self.config.initial_psi_kpa = initial_psi_kpa
        self._initialize_layers()

    def step(
        self,
        current_date: date,
        precipitation_mm: float,
        et0_mm: float,
        ndvi: Optional[float] = None,
        irrigation_mm: float = 0.0,
        dt_hours: Optional[float] = None,
    ) -> PhysicsModelOutput:
        """
        Perform one time step water balance update.

        Args:
            current_date: Current simulation date
            precipitation_mm: Precipitation for the step (mm)
            et0_mm: Reference evapotranspiration for the step (mm)
            ndvi: Normalized Difference Vegetation Index (optional)
            irrigation_mm: Irrigation applied (mm)
            dt_hours: Time step in hours (defaults to config.dt_hours)

        Returns:
            PhysicsModelOutput with updated potentials and fluxes
        """
        step_hours = dt_hours or self.config.dt_hours
        n_substeps = self._compute_substeps(step_hours)
        sub_hours = step_hours / n_substeps

        totals = {
            "infiltration_mm": 0.0,
            "runoff_mm": 0.0,
            "et_actual_mm": 0.0,
            "drainage_mm": 0.0,
            "balance_error_mm": 0.0,
        }

        for _ in range(n_substeps):
            sub = self._step_subhour(
                precipitation_mm / n_substeps,
                et0_mm / n_substeps,
                ndvi,
                irrigation_mm / n_substeps,
                sub_hours,
            )
            for key in totals:
                totals[key] += sub.get(key, 0.0)

        psi_layers = [layer.psi_kpa for layer in self.layers]
        theta_layers = [layer.theta for layer in self.layers]
        lambda_layers = [layer.lambda_phys for layer in self.layers]

        return PhysicsModelOutput(
            date=current_date,
            psi_surface_kpa=self.layers[0].psi_kpa,
            psi_root_kpa=self._get_root_zone_potential(),
            psi_deep_kpa=self.layers[-1].psi_kpa if len(
                self.layers) > 2 else None,
            precipitation_mm=precipitation_mm,
            infiltration_mm=totals["infiltration_mm"],
            runoff_mm=totals["runoff_mm"],
            evaporation_mm=0.0,
            transpiration_mm=totals["et_actual_mm"],
            drainage_mm=totals["drainage_mm"],
            water_balance_error_mm=totals["balance_error_mm"],
            converged=abs(totals["balance_error_mm"]) < 1.0,
            theta_layers=theta_layers,
            psi_layers=psi_layers,
            lambda_layers=lambda_layers,
        )

    def _step_subhour(
        self,
        precipitation_mm: float,
        et0_mm: float,
        ndvi: Optional[float],
        irrigation_mm: float,
        dt_hours: float,
    ) -> Dict[str, float]:
        """Advance a single substep and return flux diagnostics."""
        initial_storage = sum(
            layer.theta * layer.config.thickness_mm for layer in self.layers)

        total_input_mm = precipitation_mm + irrigation_mm
        top_layer = self.layers[0]
        infil_capacity = (top_layer.config.ivg_params.K_sat *
                          self.config.infiltration_factor) * (dt_hours / 24.0)
        actual_infil = min(total_input_mm, max(0.0, infil_capacity))
        runoff_mm = max(0.0, total_input_mm - actual_infil)

        n_layers = len(self.layers)
        flux_in = [0.0 for _ in range(n_layers)]
        flux_out = [0.0 for _ in range(n_layers)]
        flux_in[0] += actual_infil

        kc = self._get_crop_coefficient(ndvi)
        et_potential = et0_mm * kc

        psi_layers = [layer.psi_kpa for layer in self.layers]
        uptake = []
        for psi_kpa, root_frac in zip(psi_layers, self.config.root_fractions):
            stress = self._feddes_stress(psi_kpa)
            uptake.append(et_potential * root_frac * stress)

        et_actual = float(np.sum(uptake))

        q_interfaces = self._darcy_fluxes(psi_layers)
        dt_days = dt_hours / 24.0

        for i, q_mm_day in enumerate(q_interfaces):
            q_mm = q_mm_day * dt_days
            if q_mm >= 0:
                flux_out[i] += q_mm
                flux_in[i + 1] += q_mm
            else:
                flux_in[i] += -q_mm
                flux_out[i + 1] += -q_mm

        bottom = self.layers[-1]
        k_bottom = hydraulic_conductivity_from_psi(
            psi_layers[-1], bottom.config.ivg_params)
        drainage_mm = max(0.0, k_bottom * dt_days)
        theta_r = bottom.config.ivg_params.theta_r
        available_mm = (
            (bottom.theta - theta_r) * bottom.config.thickness_mm
            + flux_in[-1]
            - uptake[-1]
        )
        drainage_mm = min(drainage_mm, max(0.0, available_mm))
        flux_out[-1] += drainage_mm

        for i, layer in enumerate(self.layers):
            thickness_mm = layer.config.thickness_mm
            delta_theta = (flux_in[i] - flux_out[i] - uptake[i]) / thickness_mm
            new_theta = layer.theta + delta_theta
            new_theta = float(np.clip(
                new_theta, layer.config.ivg_params.theta_r, layer.config.ivg_params.theta_s))
            layer.theta = new_theta

            tparams = ThermalParams(
                bulk_density_g_cm3=layer.config.bulk_density_g_cm3,
                theta_s=layer.config.ivg_params.theta_s,
                theta_r=layer.config.ivg_params.theta_r,
                quartz_fraction=layer.config.quartz_fraction,
            )
            layer.lambda_phys = thermal_conductivity_johansen(
                layer.theta, tparams)

        final_storage = sum(
            layer.theta * layer.config.thickness_mm for layer in self.layers)
        storage_change = final_storage - initial_storage
        outputs = runoff_mm + drainage_mm + et_actual
        balance_error = storage_change - (total_input_mm - outputs)

        return {
            "infiltration_mm": actual_infil,
            "runoff_mm": runoff_mm,
            "et_actual_mm": et_actual,
            "drainage_mm": drainage_mm,
            "balance_error_mm": balance_error,
        }

    def _compute_substeps(self, dt_hours: float) -> int:
        """Compute number of substeps needed for stability."""
        min_dt = dt_hours
        for layer in self.layers:
            dz_m = layer.config.thickness_m
            k_sat_m_per_hour = layer.config.ivg_params.K_sat / 1000.0 / 24.0
            if k_sat_m_per_hour <= 0:
                continue
            dt_max = dz_m / k_sat_m_per_hour
            min_dt = min(min_dt, dt_max)

        if min_dt <= 0:
            return 1
        return max(1, int(np.ceil(dt_hours / min_dt)))

    def _get_crop_coefficient(self, ndvi: Optional[float]) -> float:
        """Get crop coefficient from NDVI or default."""
        if ndvi is None or np.isnan(ndvi):
            return self.config.default_crop_coefficient
        return 0.2 + 1.0 * np.clip((ndvi - 0.1) / 0.8, 0.0, 1.0)

    def _feddes_stress(self, psi_kpa: float) -> float:
        """Feddes-type water stress response function."""
        if psi_kpa > self.config.psi_anaerobic_kpa:
            return 0.0
        if self.config.psi_optimal_kpa < psi_kpa <= self.config.psi_anaerobic_kpa:
            return 1.0
        if self.config.psi_wilting_kpa < psi_kpa <= self.config.psi_optimal_kpa:
            num = psi_kpa - self.config.psi_wilting_kpa
            den = self.config.psi_optimal_kpa - self.config.psi_wilting_kpa
            return float(np.clip(num / den, 0.0, 1.0))
        return 0.0

    def _darcy_fluxes(self, psi_layers: List[float]) -> List[float]:
        """Compute Darcy fluxes between layers (mm/day)."""
        head_scale = 0.102
        centers = [
            0.5 * (layer.config.depth_top_m + layer.config.depth_bottom_m)
            for layer in self.layers
        ]
        q = []
        for i in range(len(self.layers) - 1):
            dz = centers[i + 1] - centers[i]
            dz = max(dz, 1e-6)
            psi_i_m = psi_layers[i] * head_scale
            psi_j_m = psi_layers[i + 1] * head_scale
            k_i = hydraulic_conductivity_from_psi(
                psi_layers[i], self.layers[i].config.ivg_params)
            k_j = hydraulic_conductivity_from_psi(
                psi_layers[i + 1], self.layers[i + 1].config.ivg_params)
            if self.config.use_geometric_mean:
                k_eff = np.sqrt(max(k_i, 0.0) * max(k_j, 0.0))
            else:
                k_eff = 0.5 * (k_i + k_j)
            gradient = (psi_i_m - psi_j_m) / dz - 1.0
            q.append(k_eff * gradient)
        return q

    def _get_root_zone_potential(self) -> float:
        """Get weighted average potential in root zone."""
        total_weight = 0.0
        weighted_psi = 0.0
        for layer, root_frac in zip(self.layers, self.config.root_fractions):
            if root_frac > 0:
                weighted_psi += layer.psi_kpa * root_frac
                total_weight += root_frac
        if total_weight > 0:
            return weighted_psi / total_weight
        return self.layers[0].psi_kpa

    def run_period(
        self,
        dates: List[date],
        precipitation: List[float],
        et0: List[float],
        ndvi: Optional[List[float]] = None,
        irrigation: Optional[List[float]] = None,
        warmup_days: int = 30,
    ) -> List[PhysicsModelOutput]:
        """
        Run model for a period of time.

        Args:
            dates: List of dates
            precipitation: Daily precipitation (mm)
            et0: Reference ET (mm)
            ndvi: NDVI values (optional)
            irrigation: Irrigation amounts (optional)
            warmup_days: Number of days for model warmup

        Returns:
            List of PhysicsModelOutput for each day
        """
        n_days = len(dates)

        # Prepare optional inputs
        if ndvi is None:
            ndvi = [None] * n_days
        if irrigation is None:
            irrigation = [0.0] * n_days

        # Run simulation
        results = []
        for i in range(n_days):
            output = self.step(
                current_date=dates[i],
                precipitation_mm=precipitation[i],
                et0_mm=et0[i],
                ndvi=ndvi[i],
                irrigation_mm=irrigation[i],
                dt_hours=24.0,
            )
            results.append(output)

        # Discard warmup period
        if warmup_days > 0 and len(results) > warmup_days:
            return results[warmup_days:]

        return results


def create_water_balance_model(
    sand_percent: float,
    clay_percent: float,
    organic_matter_percent: float = 2.0,
    n_layers: int = 3,
    max_depth_m: float = 1.0,
    initial_psi_kpa: float = -50.0,
) -> TensionSpaceWaterBalance:
    """
    Create a water balance model from soil texture.

    Args:
        sand_percent: Sand content (%)
        clay_percent: Clay content (%)
        organic_matter_percent: Organic matter (%)
        n_layers: Number of soil layers
        max_depth_m: Total soil depth (m)
        initial_psi_kpa: Initial matric potential

    Returns:
        Configured TensionSpaceWaterBalance model
    """
    # Estimate Van Genuchten parameters
    vg_params = estimate_van_genuchten_params(
        sand_percent=sand_percent,
        clay_percent=clay_percent,
        organic_matter_percent=organic_matter_percent,
    )

    # Create layer configurations
    layer_thickness = max_depth_m / n_layers
    layers = []

    for i in range(n_layers):
        depth_top = i * layer_thickness
        depth_bottom = (i + 1) * layer_thickness

        layers.append(LayerConfig(
            depth_top_m=depth_top,
            depth_bottom_m=depth_bottom,
            van_genuchten=vg_params,
        ))

    # Create config
    config = WaterBalanceConfig(
        layers=layers,
        initial_psi_kpa=initial_psi_kpa,
    )

    return TensionSpaceWaterBalance(config)

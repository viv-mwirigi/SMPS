"""
Physics Model for SMPS.

Generates physics-based priors for soil moisture prediction including:
- Water balance modeling
- Van Genuchten relationships
- ET and drainage calculations
- Physics-informed feature engineering
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

from smps.physics.van_genuchten import (
    VanGenuchtenParams, water_content_from_potential,
    potential_from_water_content, estimate_van_genuchten_params
)
from smps.physics.water_balance import (
    TensionSpaceWaterBalance,
    WaterBalanceConfig, LayerConfig
)

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConfig:
    """Configuration for physics model."""
    # Van Genuchten parameters
    use_dynamic_vg: bool = True  # Estimate VG params per site
    default_sand_pct: float = 40.0
    default_clay_pct: float = 20.0
    default_om_pct: float = 2.0

    # Water balance
    rooting_depth_m: float = 0.6
    wilting_point_psi_kpa: float = 1500.0  # -1500 kPa matric potential
    field_capacity_psi_kpa: float = 33.0   # -33 kPa matric potential

    # ET calculation
    use_weather_et: bool = True  # Use weather data ET if available
    et_method: str = 'penman_monteith'  # 'penman_monteith', 'priestley_taylor'

    # Drainage
    drainage_model: str = 'gravity'  # 'gravity', 'richards'


class PhysicsModel:
    """
    Physics-based model for generating soil moisture priors.

    Provides:
    1. Direct priors: ψ_phys (mechanistic output)
    2. Fluxes: ET_actual and drainage
    3. Plant status: K_c derived from NDVI
    4. Soil texture: Target-encoded categorical data
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        self.site_vg_params: Dict[str, VanGenuchtenParams] = {}

    def generate_physics_priors(self, df: pd.DataFrame,
                                site_manager: Optional[Any] = None) -> pd.DataFrame:
        """
        Generate physics-based priors for the entire dataframe.

        Args:
            df: Input dataframe with weather and soil data
            site_manager: Site manager for coordinate/texture data

        Returns:
            Dataframe with physics priors added
        """
        logger.info("Generating physics priors...")

        df = df.copy()

        # Get unique sites
        sites = df['station_id'].unique()

        # Generate VG parameters for each site
        for site_id in sites:
            self._get_vg_params(site_id, site_manager)

        # Process each site
        results = []
        for site_id in sites:
            site_data = df[df['station_id'] == site_id].copy()
            site_result = self._process_site_physics(site_data, site_id)
            results.append(site_result)

        # Combine results
        result_df = pd.concat(results, ignore_index=True)

        logger.info("Generated physics priors for %d sites", len(sites))
        return result_df

    def _get_vg_params(self, site_id: str, site_manager: Optional[Any]) -> VanGenuchtenParams:
        """Get Van Genuchten parameters for a site."""
        if site_id in self.site_vg_params:
            return self.site_vg_params[site_id]

        # Try to get from site manager
        if site_manager:
            site_meta = site_manager.get_site(site_id)
            if site_meta and site_meta.sand_percent is not None:
                vg = estimate_van_genuchten_params(
                    sand_percent=site_meta.sand_percent,
                    clay_percent=site_meta.clay_percent or self.config.default_clay_pct,
                    organic_matter_percent=site_meta.organic_matter_percent or self.config.default_om_pct
                )
                self.site_vg_params[site_id] = vg
                return vg

        # Use defaults
        vg = estimate_van_genuchten_params(
            sand_percent=self.config.default_sand_pct,
            clay_percent=self.config.default_clay_pct,
            organic_matter_percent=self.config.default_om_pct
        )
        self.site_vg_params[site_id] = vg
        return vg

    def _process_site_physics(self, site_df: pd.DataFrame, site_id: str) -> pd.DataFrame:
        """Process physics for a single site."""
        site_df = site_df.sort_values('date').copy()

        # Get VG parameters
        vg = self._get_vg_params(site_id, None)

        # Initialize water balance model
        # Create layer configurations
        layers = [
            LayerConfig(
                depth_top_m=0.0,
                depth_bottom_m=0.3,  # Surface layer
                van_genuchten=vg
            ),
            LayerConfig(
                depth_top_m=0.3,
                depth_bottom_m=0.6,  # Root zone layer
                van_genuchten=vg
            ),
            LayerConfig(
                depth_top_m=0.6,
                depth_bottom_m=1.0,  # Deep layer
                van_genuchten=vg
            )
        ]

        wb_config = WaterBalanceConfig(
            layers=layers,
            initial_psi_kpa=-50.0
        )

        wb_model = TensionSpaceWaterBalance(wb_config)

        # Initialize physics columns
        site_df['psi_phys_surface'] = np.nan
        site_df['psi_phys_root'] = np.nan
        site_df['psi_phys_deep'] = np.nan
        site_df['et_actual'] = np.nan
        site_df['drainage'] = np.nan
        site_df['theta_phys_surface'] = np.nan
        site_df['theta_phys_root'] = np.nan
        site_df['theta_phys_deep'] = np.nan

        # Run water balance simulation
        # Initial matric potential (kPa, negative convention)
        current_psi = -10.0
        wb_model.reset(initial_psi_kpa=current_psi)

        for idx, row in site_df.iterrows():
            try:
                # Weather inputs
                precip = row.get('precipitation_mm', 0.0)
                # Default ET if not available
                et_potential = row.get('et0_mm', 2.0)

                # Run one time step
                date_value = row.get("date")
                if hasattr(date_value, "date"):
                    date_value = date_value.date()
                result = wb_model.step(
                    current_date=date_value,
                    precipitation_mm=precip,
                    et0_mm=et_potential,
                    ndvi=row.get("ndvi") if "ndvi" in row else None,
                    irrigation_mm=0.0,
                    dt_hours=24.0,
                )

                # Store results
                site_df.at[idx, 'psi_phys_surface'] = result.psi_surface_kpa
                site_df.at[idx, 'psi_phys_root'] = result.psi_root_kpa
                site_df.at[idx, 'psi_phys_deep'] = result.psi_deep_kpa
                site_df.at[idx, 'et_actual'] = result.transpiration_mm + \
                    result.evaporation_mm
                site_df.at[idx, 'drainage'] = result.drainage_mm

                # Convert to theta for convenience
                site_df.at[idx, 'theta_phys_surface'] = water_content_from_potential(
                    result.psi_surface_kpa, vg)
                site_df.at[idx, 'theta_phys_root'] = water_content_from_potential(
                    result.psi_root_kpa, vg)
                if result.psi_deep_kpa is not None:
                    site_df.at[idx, 'theta_phys_deep'] = water_content_from_potential(
                        result.psi_deep_kpa, vg)

                # Update state
                current_psi = result.psi_surface_kpa

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(
                    "Physics calculation failed for %s at %s: %s",
                    site_id,
                    row.get("date"),
                    e,
                )
                # Use fallback values
                site_df.at[idx, 'psi_phys_surface'] = current_psi
                site_df.at[idx, 'theta_phys_surface'] = water_content_from_potential(
                    current_psi, vg)

        return site_df

    def generate_plant_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate plant status features from NDVI/satellite data.

        Args:
            df: Dataframe with satellite data

        Returns:
            Dataframe with K_c (crop coefficient) features
        """
        df = df.copy()

        # Generate K_c from NDVI if available
        if 'ndvi' in df.columns:
            # K_c typically ranges from 0.3 (bare soil) to 1.2 (full canopy)
            # NDVI ranges from -1 to 1, rescale to K_c range
            df['k_c'] = 0.3 + 0.9 * ((df['ndvi'] + 1) / 2).clip(0, 1)

            # Interpolate missing K_c values
            df['k_c'] = df.groupby('station_id')['k_c'].apply(
                lambda x: x.interpolate(method='cubic', limit_direction='both')
            ).fillna(0.6)  # Default K_c for mixed vegetation

        else:
            # Default K_c if no NDVI available
            df['k_c'] = 0.6

        # Generate K_c features
        df['k_c_smoothed'] = df.groupby('station_id')['k_c'].transform(
            lambda x: x.rolling(window=7, center=True, min_periods=1).mean()
        ).fillna(df['k_c'])

        df['k_c_trend'] = df.groupby('station_id')['k_c'].transform(
            lambda x: x.diff(7)
        ).fillna(0)

        return df

    def generate_soil_texture_features(self, df: pd.DataFrame,
                                       site_manager: Optional[Any] = None) -> pd.DataFrame:
        """
        Generate soil texture features.

        Args:
            df: Input dataframe
            site_manager: Site manager for texture data

        Returns:
            Dataframe with soil texture features
        """
        df = df.copy()

        # Add texture features from site manager
        if site_manager:
            df['sand_percent'] = df['station_id'].apply(
                lambda x: site_manager.get_site(x).sand_percent if site_manager.get_site(
                    x) else self.config.default_sand_pct
            )
            df['clay_percent'] = df['station_id'].apply(
                lambda x: site_manager.get_site(x).clay_percent if site_manager.get_site(
                    x) else self.config.default_clay_pct
            )
            df['soil_texture_class'] = df['station_id'].apply(
                lambda x: site_manager.get_site(
                    x).soil_texture if site_manager.get_site(x) else 'loam'
            )
        else:
            # Use defaults
            df['sand_percent'] = self.config.default_sand_pct
            df['clay_percent'] = self.config.default_clay_pct
            df['soil_texture_class'] = 'loam'

        # Create texture-based features
        df['texture_index'] = (df['sand_percent'] -
                               df['clay_percent']) / 100.0  # -1 to 1
        df['bulk_density_proxy'] = 1.65 - 0.318 * \
            (df['sand_percent'] / 100.0)  # g/cm³

        return df

    def convert_targets_for_training(self, df: pd.DataFrame,
                                     target_space: str = 'theta') -> pd.DataFrame:
        """
        Convert targets to appropriate space for training.

        Args:
            df: Dataframe with observed soil moisture
            target_space: 'theta' or 'psi'

        Returns:
            Dataframe with converted targets
        """
        df = df.copy()

        if target_space == 'theta':
            # Keep theta as-is (assuming observations are in theta)
            df['target'] = df['soil_moisture']
            df['physics_prior'] = df['theta_phys_surface']

        elif target_space == 'psi':
            # Convert observed theta to psi for training
            for site_id in df['station_id'].unique():
                site_mask = df['station_id'] == site_id
                vg = self._get_vg_params(site_id, None)

                # Convert observed theta to psi
                theta_obs = df.loc[site_mask, 'soil_moisture'].values
                psi_obs = np.array(
                    [potential_from_water_content(t, vg) for t in theta_obs])

                df.loc[site_mask, 'target'] = psi_obs
                df.loc[site_mask, 'physics_prior'] = df.loc[site_mask,
                                                            'psi_phys_surface']

        else:
            raise ValueError(f"Unknown target space: {target_space}")

        return df

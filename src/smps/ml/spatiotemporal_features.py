"""
Spatiotemporal Feature Engineering for Soil Moisture Modeling.

Addresses the critical limitation of station-wise independent regression by adding:
1. Spatial correlation features (distance-weighted interpolation)
2. Regional climate coherence
3. Spatial attention mechanisms
4. Watershed connectivity features

This transforms N independent time series into a spatiotemporal field model.

Key Concepts:
- Spatial autocorrelation: Nearby stations influence each other
- Regional climate patterns: Weather systems affect multiple sites
- Watershed connectivity: Water flow between locations
- Spatial covariates: Continuous fields (elevation, soil properties)

References:
- Hengl et al. (2017): Soil spatial prediction using ML
- McBratney et al. (2003): Pedometrics and spatial soil prediction
- Goovaerts (1997): Geostatistics for natural resources evaluation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SpatialConfig:
    """Configuration for spatiotemporal feature engineering."""

    # Spatial interpolation parameters
    max_neighbor_distance_km: float = 50.0  # Maximum distance for neighbors
    min_neighbors: int = 3  # Minimum neighbors for interpolation
    max_neighbors: int = 10  # Maximum neighbors to consider

    # Distance weighting parameters
    distance_power: float = 2.0  # Power for inverse distance weighting
    spatial_decay_factor: float = 0.1  # Exponential decay factor

    # Regional climate parameters
    regional_radius_km: float = 100.0  # Radius for regional features
    climate_variables: List[str] = field(default_factory=lambda: [
        'precipitation_mm', 'temperature_mean_c', 'et0_mm',
        'relative_humidity_mean', 'wind_speed_mean_m_s'
    ])

    # Spatial attention parameters
    attention_heads: int = 4
    attention_dim: int = 32
    spatial_attention_radius: float = 25.0  # km

    # Topography and terrain parameters
    include_topography: bool = True
    topography_variables: List[str] = field(default_factory=lambda: [
        'elevation_m', 'slope_degrees', 'aspect_degrees', 'twi',
        'curvature', 'hillshade'
    ])

    # Land use and land cover parameters
    include_land_use: bool = True
    land_use_variables: List[str] = field(default_factory=lambda: [
        'land_cover_class', 'crop_type', 'irrigation_status',
        'urban_proximity_km', 'water_body_distance_km'
    ])

    # Watershed and hydrology parameters
    include_watershed: bool = True
    watershed_variables: List[str] = field(default_factory=lambda: [
        'watershed_area_km2', 'stream_order', 'drainage_density',
        'flow_accumulation', 'wetness_index'
    ])

    # Soil spatial parameters
    include_soil_spatial: bool = True
    soil_spatial_variables: List[str] = field(default_factory=lambda: [
        'soil_type', 'soil_texture_class', 'soil_depth_cm',
        'parent_material', 'geology_class'
    ])


class SpatialFeatureEngineer:
    """
    Engineer spatiotemporal features to capture spatial correlations.

    This addresses the fundamental limitation of treating soil moisture as
    N independent time series rather than a spatiotemporal field.
    """

    def __init__(self, config: SpatialConfig = None):
        self.config = config or SpatialConfig()
        self.scaler = StandardScaler()

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in km."""
        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def find_spatial_neighbors(
        self,
        target_lat: float,
        target_lon: float,
        all_sites: Dict[str, Tuple[float, float]],
        max_distance_km: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find spatially nearby sites with distances.

        Args:
            target_lat: Target site latitude
            target_lon: Target site longitude
            all_sites: Dict mapping site_id to (lat, lon)
            max_distance_km: Maximum distance to consider

        Returns:
            List of (site_id, distance_km) tuples, sorted by distance
        """
        if max_distance_km is None:
            max_distance_km = self.config.max_neighbor_distance_km

        neighbors = []
        for site_id, (lat, lon) in all_sites.items():
            if lat == target_lat and lon == target_lon:
                continue  # Skip self

            distance = self.haversine_distance(
                target_lat, target_lon, lat, lon)
            if distance <= max_distance_km:
                neighbors.append((site_id, distance))

        # Sort by distance
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:self.config.max_neighbors]

    def inverse_distance_weighting(
        self,
        values: np.ndarray,
        distances: np.ndarray,
        power: Optional[float] = None
    ) -> float:
        """
        Calculate inverse distance weighted average.

        Args:
            values: Array of values from neighboring sites
            distances: Array of distances to neighboring sites
            power: Power for inverse distance weighting

        Returns:
            Weighted average value
        """
        if power is None:
            power = self.config.distance_power

        # Avoid division by zero
        distances = np.maximum(distances, 0.1)

        # Calculate weights
        weights = 1.0 / (distances ** power)
        weights = weights / np.sum(weights)  # Normalize

        return np.sum(values * weights)

    def add_spatial_interpolation_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict],
        target_variables: List[str]
    ) -> pd.DataFrame:
        """
        Add spatial interpolation features using nearby stations.

        For each target variable, adds features like:
        - spatial_mean_{var}: Distance-weighted mean from neighbors
        - spatial_std_{var}: Standard deviation from neighbors
        - spatial_gradient_{var}: Spatial gradient estimate

        Args:
            df: DataFrame with site_id, date, and target variables
            site_metadata: Dict mapping site_id to metadata including lat/lon
            target_variables: Variables to spatially interpolate

        Returns:
            DataFrame with added spatial features
        """
        df = df.copy()

        # Extract site locations
        site_locations = {}
        for site_id, metadata in site_metadata.items():
            if 'latitude' in metadata and 'longitude' in metadata:
                site_locations[site_id] = (
                    metadata['latitude'], metadata['longitude'])

        # Group by date to process each time step
        for date_val in df['date'].unique():
            date_mask = df['date'] == date_val

            for target_var in target_variables:
                if target_var not in df.columns:
                    continue

                # Get values for this date
                date_data = df[date_mask].copy()

                for idx, row in date_data.iterrows():
                    site_id = row['site_id']
                    if site_id not in site_locations:
                        continue

                    target_lat, target_lon = site_locations[site_id]

                    # Find neighbors
                    neighbors = self.find_spatial_neighbors(
                        target_lat, target_lon, site_locations
                    )

                    if len(neighbors) < self.config.min_neighbors:
                        continue

                    # Get neighbor values
                    neighbor_values = []
                    distances = []

                    for neighbor_id, distance in neighbors:
                        neighbor_row = date_data[date_data['site_id']
                                                 == neighbor_id]
                        if not neighbor_row.empty and not pd.isna(neighbor_row[target_var].iloc[0]):
                            neighbor_values.append(
                                neighbor_row[target_var].iloc[0])
                            distances.append(distance)

                    if len(neighbor_values) >= self.config.min_neighbors:
                        neighbor_values = np.array(neighbor_values)
                        distances = np.array(distances)

                        # Calculate spatial features
                        spatial_mean = self.inverse_distance_weighting(
                            neighbor_values, distances
                        )
                        spatial_std = np.std(neighbor_values)
                        spatial_gradient = np.mean(
                            np.abs(neighbor_values - neighbor_values.mean()))

                        # Add to dataframe
                        df.loc[idx,
                               f'spatial_mean_{target_var}'] = spatial_mean
                        df.loc[idx, f'spatial_std_{target_var}'] = spatial_std
                        df.loc[idx,
                               f'spatial_gradient_{target_var}'] = spatial_gradient

        return df

    def add_regional_climate_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Add regional climate coherence features.

        Calculates regional averages for climate variables within a radius,
        capturing mesoscale weather patterns that affect multiple sites.

        Args:
            df: DataFrame with climate variables
            site_metadata: Site metadata with lat/lon

        Returns:
            DataFrame with regional climate features
        """
        df = df.copy()

        # Extract site locations
        site_locations = {}
        for site_id, metadata in site_metadata.items():
            if 'latitude' in metadata and 'longitude' in metadata:
                site_locations[site_id] = (
                    metadata['latitude'], metadata['longitude'])

        # Group by date
        for date_val in df['date'].unique():
            date_mask = df['date'] == date_val
            date_data = df[date_mask].copy()

            for idx, row in date_data.iterrows():
                site_id = row['site_id']
                if site_id not in site_locations:
                    continue

                target_lat, target_lon = site_locations[site_id]

                # Find sites within regional radius
                regional_sites = self.find_spatial_neighbors(
                    target_lat, target_lon, site_locations,
                    max_distance_km=self.config.regional_radius_km
                )

                for var in self.config.climate_variables:
                    if var not in df.columns:
                        continue

                    # Get regional values
                    regional_values = []
                    for neighbor_id, distance in regional_sites:
                        neighbor_row = date_data[date_data['site_id']
                                                 == neighbor_id]
                        if not neighbor_row.empty and not pd.isna(neighbor_row[var].iloc[0]):
                            regional_values.append(neighbor_row[var].iloc[0])

                    if len(regional_values) >= 3:  # Need minimum for regional average
                        regional_mean = np.mean(regional_values)
                        regional_std = np.std(regional_values)

                        df.loc[idx, f'regional_mean_{var}'] = regional_mean
                        df.loc[idx, f'regional_std_{var}'] = regional_std

        return df

    def add_spatial_covariates(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Add comprehensive spatial covariate features.

        Includes topography, land use, land cover, and soil spatial features
        that influence soil moisture patterns.

        Args:
            df: DataFrame with site data
            site_metadata: Site metadata

        Returns:
            DataFrame with spatial covariates
        """
        df = df.copy()

        # Add topography features
        if self.config.include_topography:
            df = self._add_topography_features(df, site_metadata)

        # Add land use and land cover features
        if self.config.include_land_use:
            df = self._add_land_use_features(df, site_metadata)

        # Add soil spatial features
        if self.config.include_soil_spatial:
            df = self._add_soil_spatial_features(df, site_metadata)

        return df

    def _add_topography_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Add topographic and terrain features."""
        df = df.copy()

        # Extract topographic data for all sites
        topo_data = {}
        for site_id, metadata in site_metadata.items():
            topo_data[site_id] = {}
            for var in self.config.topography_variables:
                if var in metadata:
                    topo_data[site_id][var] = metadata[var]
                else:
                    # Default values for missing data
                    defaults = {
                        'elevation_m': 100.0,
                        'slope_degrees': 5.0,
                        'aspect_degrees': 180.0,
                        'twi': 5.0,
                        'curvature': 0.0,
                        'hillshade': 0.5
                    }
                    topo_data[site_id][var] = defaults.get(var, 0.0)

        # Add topographic features to each site
        for site_id in df['site_id'].unique():
            if site_id in topo_data:
                site_mask = df['site_id'] == site_id
                topo = topo_data[site_id]

                # Direct topographic features
                for var in self.config.topography_variables:
                    df.loc[site_mask, f'topo_{var}'] = topo[var]

                # Derived topographic features
                if 'slope_degrees' in topo:
                    # Slope radiation factor (affects ET)
                    slope_rad = np.sin(np.radians(topo['slope_degrees']))
                    df.loc[site_mask, 'topo_slope_radiation_factor'] = slope_rad

                if 'elevation_m' in topo:
                    # Elevation-based climate proxy
                    df.loc[site_mask, 'topo_elevation_zone'] = self._categorize_elevation(
                        topo['elevation_m'])

                # Topographic wetness potential
                if 'twi' in topo and 'slope_degrees' in topo:
                    wetness_potential = topo['twi'] / \
                        (topo['slope_degrees'] + 1)
                    df.loc[site_mask, 'topo_wetness_potential'] = wetness_potential

        # Regional topographic statistics
        df = self._add_regional_topography_stats(df, topo_data, site_metadata)

        return df

    def _add_land_use_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Add land use and land cover features."""
        df = df.copy()

        # Extract land use data
        land_use_data = {}
        for site_id, metadata in site_metadata.items():
            land_use_data[site_id] = {}
            for var in self.config.land_use_variables:
                if var in metadata:
                    land_use_data[site_id][var] = metadata[var]
                else:
                    # Default values
                    defaults = {
                        'land_cover_class': 'unknown',
                        'crop_type': 'unknown',
                        'irrigation_status': 'unknown',
                        'urban_proximity_km': 10.0,
                        'water_body_distance_km': 5.0
                    }
                    land_use_data[site_id][var] = defaults.get(var, 'unknown')

        # Add land use features
        for site_id in df['site_id'].unique():
            if site_id in land_use_data:
                site_mask = df['site_id'] == site_id
                land_use = land_use_data[site_id]

                # Direct land use features
                for var in self.config.land_use_variables:
                    value = land_use[var]
                    if isinstance(value, str):
                        # One-hot encode categorical variables
                        df.loc[site_mask, f'landuse_{var}_{value}'] = 1
                    else:
                        # Numeric variables
                        df.loc[site_mask, f'landuse_{var}'] = value

                # Derived land use features
                if 'irrigation_status' in land_use:
                    irrigation_type = land_use['irrigation_status']
                    if irrigation_type == 'irrigated':
                        df.loc[site_mask, 'landuse_irrigated'] = 1
                        df.loc[site_mask, 'landuse_water_limited'] = 0
                    else:
                        df.loc[site_mask, 'landuse_irrigated'] = 0
                        df.loc[site_mask, 'landuse_water_limited'] = 1

                # Urban heat island effect proxy
                if 'urban_proximity_km' in land_use:
                    urban_effect = 1 / (land_use['urban_proximity_km'] + 1)
                    df.loc[site_mask, 'landuse_urban_heat_effect'] = urban_effect

        return df

    def _add_soil_spatial_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Add soil spatial variability features."""
        df = df.copy()

        # Extract soil spatial data
        soil_data = {}
        for site_id, metadata in site_metadata.items():
            soil_data[site_id] = {}
            for var in self.config.soil_spatial_variables:
                if var in metadata:
                    soil_data[site_id][var] = metadata[var]
                else:
                    # Default values
                    defaults = {
                        'soil_type': 'unknown',
                        'soil_texture_class': 'loam',
                        'soil_depth_cm': 100.0,
                        'parent_material': 'unknown',
                        'geology_class': 'unknown'
                    }
                    soil_data[site_id][var] = defaults.get(var, 'unknown')

        # Add soil spatial features
        for site_id in df['site_id'].unique():
            if site_id in soil_data:
                site_mask = df['site_id'] == site_id
                soil = soil_data[site_id]

                # Direct soil features
                for var in self.config.soil_spatial_variables:
                    value = soil[var]
                    if isinstance(value, str):
                        # One-hot encode categorical variables
                        df.loc[site_mask, f'soil_{var}_{value}'] = 1
                    else:
                        # Numeric variables
                        df.loc[site_mask, f'soil_{var}'] = value

                # Derived soil features
                if 'soil_texture_class' in soil:
                    texture = soil['soil_texture_class']
                    # Texture drainage classes (1=very slow, 5=very fast)
                    drainage_classes = {
                        'clay': 1, 'silty_clay': 1, 'clay_loam': 2,
                        'loam': 3, 'sandy_loam': 4, 'loamy_sand': 5, 'sand': 5
                    }
                    drainage_class = drainage_classes.get(texture, 3)
                    df.loc[site_mask, 'soil_drainage_class'] = drainage_class

        return df

    def _add_regional_topography_stats(
        self,
        df: pd.DataFrame,
        topo_data: Dict[str, Dict],
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Add regional topographic statistics."""
        df = df.copy()

        # Calculate regional statistics for each site
        for site_id in df['site_id'].unique():
            if site_id not in topo_data:
                continue

            site_mask = df['site_id'] == site_id
            site_topo = topo_data[site_id]

            # Get site lat/lon from metadata
            meta = site_metadata.get(site_id, {})
            site_lat = meta.get('latitude')
            site_lon = meta.get('longitude')
            if site_lat is None or site_lon is None:
                # Can't compute regional stats without location
                continue

            # Find nearby sites within regional radius
            regional_elevs = []
            for other_id, other_meta in site_metadata.items():
                if other_id == site_id:
                    continue
                other_lat = other_meta.get('latitude')
                other_lon = other_meta.get('longitude')
                if other_lat is None or other_lon is None:
                    continue
                dist = self.haversine_distance(
                    site_lat, site_lon, other_lat, other_lon)
                if dist <= self.config.regional_radius_km:
                    # Prefer topo_data elevation if available, else metadata
                    if other_id in topo_data:
                        elev = topo_data[other_id].get(
                            'elevation_m', other_meta.get('elevation_m'))
                    else:
                        elev = other_meta.get('elevation_m')
                    if elev is not None:
                        regional_elevs.append(elev)

            if regional_elevs:
                df.loc[site_mask, 'regional_elev_mean'] = np.mean(
                    regional_elevs)
                df.loc[site_mask, 'regional_elev_std'] = np.std(regional_elevs)
                df.loc[site_mask, 'regional_elev_range'] = np.max(
                    regional_elevs) - np.min(regional_elevs)

        return df

    def _categorize_elevation(self, elevation_m: float) -> int:
        """Categorize elevation into zones."""
        if elevation_m < 200:
            return 1  # Lowland
        elif elevation_m < 500:
            return 2  # Midland
        elif elevation_m < 1000:
            return 3  # Highland
        else:
            return 4  # Mountain

    def add_watershed_connectivity_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Add watershed connectivity features.

        Models how water flows between locations based on topography:
        - Downstream connectivity (water flow direction)
        - Upslope contributing area
        - Flow accumulation patterns

        Args:
            df: DataFrame with site data
            site_metadata: Site metadata with elevation and slope

        Returns:
            DataFrame with watershed features
        """
        df = df.copy()

        # Extract topographic data
        topo_data = {}
        for site_id, metadata in site_metadata.items():
            if all(k in metadata for k in ['latitude', 'longitude', 'elevation_m', 'slope_percent']):
                topo_data[site_id] = {
                    'lat': metadata['latitude'],
                    'lon': metadata['longitude'],
                    'elev': metadata['elevation_m'],
                    'slope': metadata['slope_percent']
                }

        # Calculate watershed features for each site
        for site_id, topo in topo_data.items():
            site_mask = df['site_id'] == site_id

            # Find downstream sites (lower elevation within flow distance)
            downstream_sites = []
            for other_id, other_topo in topo_data.items():
                if other_id == site_id:
                    continue

                distance = self.haversine_distance(
                    topo['lat'], topo['lon'], other_topo['lat'], other_topo['lon']
                )

                # Simple flow direction estimate based on elevation difference
                elev_diff = topo['elev'] - other_topo['elev']
                flow_distance = distance * \
                    topo['slope'] / 100.0  # Rough flow distance

                if elev_diff > 0 and flow_distance < 10.0:  # Within flow distance
                    downstream_sites.append((other_id, distance, elev_diff))

            if downstream_sites:
                # Sort by elevation difference (steepest drop first)
                downstream_sites.sort(key=lambda x: x[2], reverse=True)

                # Calculate connectivity metrics
                nearest_downstream_dist = downstream_sites[0][1]
                mean_downstream_elev_diff = np.mean(
                    [x[2] for x in downstream_sites])

                df.loc[site_mask, 'downstream_distance_km'] = nearest_downstream_dist
                df.loc[site_mask,
                       'mean_downstream_elev_diff_m'] = mean_downstream_elev_diff
                df.loc[site_mask, 'downstream_sites_count'] = len(
                    downstream_sites)

        return df

    def add_irrigation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add irrigation management and scheduling features.

        Critical for agricultural soil moisture prediction where irrigation
        is a major water input that the model must account for.
        """
        logger.info("Engineering irrigation features...")

        # Check if irrigation data exists
        irrigation_cols = [
            col for col in df.columns if 'irrigation' in col.lower()]
        if not irrigation_cols:
            logger.warning(
                "No irrigation data found, skipping irrigation features")
            return df

        # Use the first irrigation column found (typically 'irrigation_mm')
        irrigation_col = irrigation_cols[0]

        # Irrigation frequency and timing features
        df['irrigation_days_since_last'] = self._calculate_days_since_last_irrigation(
            df, irrigation_col)
        df['irrigation_frequency_7d'] = self._calculate_irrigation_frequency(
            df, irrigation_col, 7)
        df['irrigation_frequency_30d'] = self._calculate_irrigation_frequency(
            df, irrigation_col, 30)

        # Irrigation amount features
        df['irrigation_cumulative_7d'] = df[irrigation_col].rolling(
            7, min_periods=1).sum()
        df['irrigation_cumulative_30d'] = df[irrigation_col].rolling(
            30, min_periods=1).sum()
        df['irrigation_avg_amount'] = df[irrigation_col].rolling(
            30, min_periods=1).mean()

        # Irrigation scheduling patterns
        df['irrigation_regularity_score'] = self._calculate_irrigation_regularity(
            df, irrigation_col)
        df['irrigation_dry_spell_relief'] = self._calculate_dry_spell_relief(
            df, irrigation_col)

        # Irrigation efficiency and management features
        df['irrigation_water_use_efficiency'] = self._calculate_water_use_efficiency(
            df, irrigation_col)
        df['irrigation_timing_score'] = self._calculate_irrigation_timing_score(
            df, irrigation_col)

        # Irrigation-soil moisture interactions
        sm_cols = [col for col in df.columns if 'sm' in col.lower()
                   and 'obs' in col.lower()]
        if sm_cols:
            sm_col = sm_cols[0]
            df['irrigation_impact_on_sm'] = self._calculate_irrigation_impact(
                df, irrigation_col, sm_col)

        logger.info(
            f"Added {len([col for col in df.columns if col.startswith('irrigation_')])} irrigation features")
        return df

    def _calculate_days_since_last_irrigation(self, df: pd.DataFrame, irrigation_col: str) -> pd.Series:
        """Calculate days since last irrigation event."""
        irrigation_mask = df[irrigation_col] > 0
        days_since = pd.Series(index=df.index, dtype=float)

        last_irrigation_idx = None
        for i, (idx, row) in enumerate(df.iterrows()):
            if row[irrigation_col] > 0:
                last_irrigation_idx = i
                days_since.iloc[i] = 0
            elif last_irrigation_idx is not None:
                days_since.iloc[i] = i - last_irrigation_idx
            else:
                days_since.iloc[i] = np.nan  # No prior irrigation

        return days_since

    def _calculate_irrigation_frequency(self, df: pd.DataFrame, irrigation_col: str, window: int) -> pd.Series:
        """Calculate irrigation frequency over a rolling window."""
        irrigation_events = (df[irrigation_col] > 0).astype(int)
        return irrigation_events.rolling(window, min_periods=1).sum()

    def _calculate_irrigation_regularity(self, df: pd.DataFrame, irrigation_col: str) -> pd.Series:
        """Calculate how regular irrigation scheduling is."""
        irrigation_events = (df[irrigation_col] > 0)
        if irrigation_events.sum() < 3:
            return pd.Series(0.0, index=df.index)

        # Calculate coefficient of variation of irrigation intervals
        irrigation_dates = df.index[irrigation_events]
        if len(irrigation_dates) < 2:
            return pd.Series(0.5, index=df.index)  # Neutral score

        intervals = np.diff(irrigation_dates)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval == 0:
            return pd.Series(1.0, index=df.index)  # Perfect regularity (daily)

        cv = std_interval / mean_interval  # Coefficient of variation
        # Convert to 0-1 score (higher is more regular)
        regularity = 1.0 / (1.0 + cv)

        return pd.Series(regularity, index=df.index)

    def _calculate_dry_spell_relief(self, df: pd.DataFrame, irrigation_col: str) -> pd.Series:
        """Calculate how irrigation relieves dry spells."""
        # This would require precipitation data to identify dry spells
        # For now, return a placeholder
        return pd.Series(0.0, index=df.index)

    def _calculate_water_use_efficiency(self, df: pd.DataFrame, irrigation_col: str) -> pd.Series:
        """Calculate irrigation water use efficiency."""
        # Efficiency based on irrigation amount relative to typical needs
        # This is a simplified metric
        rolling_avg = df[irrigation_col].rolling(30, min_periods=1).mean()
        # Higher irrigation = lower efficiency
        efficiency = 1.0 / (1.0 + rolling_avg / 10.0)
        return efficiency.clip(0, 1)

    def _calculate_irrigation_timing_score(self, df: pd.DataFrame, irrigation_col: str) -> pd.Series:
        """Calculate irrigation timing optimality score."""
        # Simplified: score based on whether irrigation happens during dry periods
        # This would be improved with weather data
        return pd.Series(0.5, index=df.index)  # Neutral score

    def _calculate_irrigation_impact(self, df: pd.DataFrame, irrigation_col: str, sm_col: str) -> pd.Series:
        """Calculate irrigation impact on soil moisture."""
        # Calculate soil moisture change following irrigation
        sm_change = df[sm_col].diff()
        irrigation_impact = pd.Series(0.0, index=df.index)

        # Look for soil moisture increases following irrigation
        irrigation_mask = df[irrigation_col] > 0
        for i in range(1, len(df)):
            if irrigation_mask.iloc[i-1] and not pd.isna(sm_change.iloc[i]):
                irrigation_impact.iloc[i] = sm_change.iloc[i]

        return irrigation_impact

    def engineer_spatiotemporal_features(
        self,
        df: pd.DataFrame,
        site_metadata: Dict[str, Dict],
        target_variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply all spatiotemporal feature engineering.

        Args:
            df: Input DataFrame
            site_metadata: Site metadata dictionary
            target_variables: Variables to spatially interpolate

        Returns:
            DataFrame with spatiotemporal features
        """
        if target_variables is None:
            target_variables = ['obs_sm_10cm',
                                'physics_sm_surface', 'physics_sm_root']

        logger.info("Adding spatial interpolation features...")
        df = self.add_spatial_interpolation_features(
            df, site_metadata, target_variables)

        logger.info("Adding regional climate features...")
        df = self.add_regional_climate_features(df, site_metadata)

        logger.info("Adding spatial covariates...")
        df = self.add_spatial_covariates(df, site_metadata)

        logger.info("Adding watershed connectivity features...")
        df = self.add_watershed_connectivity_features(df, site_metadata)

        logger.info("Adding irrigation management features...")
        df = self.add_irrigation_features(df)

        # Fill NaN values in new features
        spatial_cols = [col for col in df.columns if col.startswith(
            ('spatial_', 'regional_', 'downstream_', 'irrigation_'))]
        for col in spatial_cols:
            df[col] = df[col].fillna(df[col].mean())

        logger.info(f"Added {len(spatial_cols)} spatiotemporal features")
        return df

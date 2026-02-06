#!/usr/bin/env python3
"""
ML-Assisted Physics Parameter Estimation.

Uses hybrid model residuals to diagnose physics model deficiencies
and derive parameter corrections for different soil-climate regimes.

This implements the workflow:
1. Analyze residual patterns to identify systematic physics errors
2. Cluster sites by soil-climate regime
3. Train interpretable models to explain residuals
4. Convert insights into physics parameter corrections
5. Generate updated calibration parameters

Theory:
    residual(t,s) = ML_hybrid(t,s) - Physics(t,s)

    If residuals are systematically positive → physics too dry
    If residuals are systematically negative → physics too wet

    Pattern analysis reveals which parameters need adjustment:
    - High residuals when rain high + sand high → Ksat too high
    - High residuals when SM low + ET high → Missing ET stress function
    - Consistent positive bias → Porosity/FC too low

References:
    - Data assimilation in land surface models
    - Parameter estimation via machine learning
    - Pedotransfer function calibration
"""

import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER CORRECTION DATACLASS
# =============================================================================

@dataclass
class PhysicsCorrection:
    """Parameter correction derived from residual analysis."""
    parameter_name: str
    correction_type: str  # 'multiplicative', 'additive', 'threshold'
    correction_value: float
    confidence: float  # 0-1
    applicable_conditions: Dict[str, Tuple[float, float]]  # Feature ranges
    rationale: str


@dataclass
class ClusterProfile:
    """Profile for a soil-climate cluster."""
    cluster_id: int
    n_sites: int
    n_samples: int

    # Cluster characteristics
    mean_clay: float
    mean_sand: float
    mean_precip: float
    mean_et0: float
    climate_zone: str

    # Residual statistics
    mean_residual: float
    std_residual: float
    residual_skew: float

    # Derived corrections
    corrections: List[PhysicsCorrection]


# =============================================================================
# RESIDUAL ANALYSIS
# =============================================================================

class ResidualDiagnostics:
    """
    Analyze residuals to diagnose physics model deficiencies.
    """

    def __init__(self, canonical_df: pd.DataFrame):
        """
        Initialize with canonical table containing physics predictions and observations.

        Args:
            canonical_df: DataFrame with columns including:
                - soil_moisture (observed)
                - physics_prior (physics prediction)
                - residual (observed - physics)
                - Weather: precipitation_mm, et0_mm, temperature_*
                - Soil: clay_pct, sand_pct, organic_carbon_pct
                - Location: latitude, longitude, elevation_m
        """
        self.df = canonical_df.copy()
        self._validate_columns()
        self._compute_derived_features()

    def _validate_columns(self):
        """Ensure required columns exist."""
        required = ['soil_moisture', 'physics_prior', 'residual',
                    'precipitation_mm', 'et0_mm', 'clay_pct', 'sand_pct']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _compute_derived_features(self):
        """Compute additional features for analysis."""
        df = self.df

        # Normalized residual (relative error)
        df['residual_relative'] = df['residual'] / (df['soil_moisture'] + 0.01)

        # Physics state categories
        df['physics_dry'] = (df['physics_prior'] < 0.15).astype(int)
        df['physics_wet'] = (df['physics_prior'] > 0.35).astype(int)

        # Weather intensity
        df['rain_intensity'] = pd.cut(df['precipitation_mm'],
                                      bins=[-np.inf, 0.1, 5, 20, np.inf],
                                      labels=['none', 'light', 'moderate', 'heavy'])
        df['et_intensity'] = pd.cut(df['et0_mm'],
                                    bins=[-np.inf, 2, 4, 6, np.inf],
                                    labels=['low', 'moderate', 'high', 'very_high'])

        # Soil texture class
        df['texture_class'] = df.apply(self._classify_texture, axis=1)

        # Water balance
        if 'water_balance_1d' not in df.columns:
            df['water_balance_1d'] = df['precipitation_mm'] - df['et0_mm']

    def _classify_texture(self, row) -> str:
        """Classify soil texture from sand/clay."""
        sand, clay = row['sand_pct'], row['clay_pct']
        if sand > 70:
            return 'sandy'
        elif clay > 40:
            return 'clayey'
        elif sand > 50 and clay < 20:
            return 'sandy_loam'
        elif clay > 25:
            return 'clay_loam'
        else:
            return 'loam'

    def compute_residual_statistics(self) -> pd.DataFrame:
        """Compute residual statistics by various groupings."""
        df = self.df

        results = []

        # Overall statistics
        results.append({
            'group': 'overall',
            'subgroup': 'all',
            'n_samples': len(df),
            'mean_residual': df['residual'].mean(),
            'std_residual': df['residual'].std(),
            'median_residual': df['residual'].median(),
            'rmse_physics': np.sqrt((df['residual']**2).mean()),
            'bias_direction': 'dry' if df['residual'].mean() > 0 else 'wet'
        })

        # By texture class
        for texture in df['texture_class'].unique():
            mask = df['texture_class'] == texture
            sub = df[mask]
            if len(sub) > 100:
                results.append({
                    'group': 'texture',
                    'subgroup': texture,
                    'n_samples': len(sub),
                    'mean_residual': sub['residual'].mean(),
                    'std_residual': sub['residual'].std(),
                    'median_residual': sub['residual'].median(),
                    'rmse_physics': np.sqrt((sub['residual']**2).mean()),
                    'bias_direction': 'dry' if sub['residual'].mean() > 0 else 'wet'
                })

        # By rain intensity
        for intensity in ['none', 'light', 'moderate', 'heavy']:
            mask = df['rain_intensity'] == intensity
            sub = df[mask]
            if len(sub) > 100:
                results.append({
                    'group': 'rain',
                    'subgroup': intensity,
                    'n_samples': len(sub),
                    'mean_residual': sub['residual'].mean(),
                    'std_residual': sub['residual'].std(),
                    'median_residual': sub['residual'].median(),
                    'rmse_physics': np.sqrt((sub['residual']**2).mean()),
                    'bias_direction': 'dry' if sub['residual'].mean() > 0 else 'wet'
                })

        # By physics state
        for state, label in [('physics_dry', 'dry_state'), ('physics_wet', 'wet_state')]:
            mask = df[state] == 1
            sub = df[mask]
            if len(sub) > 100:
                results.append({
                    'group': 'physics_state',
                    'subgroup': label,
                    'n_samples': len(sub),
                    'mean_residual': sub['residual'].mean(),
                    'std_residual': sub['residual'].std(),
                    'median_residual': sub['residual'].median(),
                    'rmse_physics': np.sqrt((sub['residual']**2).mean()),
                    'bias_direction': 'dry' if sub['residual'].mean() > 0 else 'wet'
                })

        return pd.DataFrame(results)

    def train_residual_explainer(self) -> Tuple[lgb.Booster, pd.DataFrame]:
        """
        Train an interpretable model to explain residuals.

        This model reveals which physical conditions drive residuals,
        which directly informs parameter corrections.

        Returns:
            model: Trained LightGBM model
            importance: Feature importance DataFrame
        """
        df = self.df.dropna(subset=['residual'])

        # Select features that can inform physics corrections
        # These are the INPUTS to the physics model, not the outputs
        feature_cols = [
            # Weather forcings
            'precipitation_mm', 'et0_mm',
            'temperature_mean_c', 'temperature_max_c', 'temperature_min_c',
            # Soil properties (PTF inputs)
            'clay_pct', 'sand_pct', 'organic_carbon_pct',
            # Site characteristics
            'latitude', 'elevation_m',
            # Derived
            'water_balance_1d',
        ]

        # Add seasonal features if available
        for col in ['day_of_year', 'month']:
            if col in df.columns:
                feature_cols.append(col)

        # Add saturation if available
        if 'saturation' in df.columns:
            feature_cols.append('saturation')

        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].copy()
        y = df['residual'].values

        # Handle missing values
        X = X.fillna(X.median())

        # Train LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # Keep simple for interpretability
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
            'n_jobs': -1,
        }

        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=200)

        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        return model, importance

    def analyze_interactions(self) -> pd.DataFrame:
        """
        Analyze key interactions that drive residuals.

        Returns DataFrame with interaction effects on residuals.
        """
        df = self.df

        interactions = []

        # Rain × Sand interaction (infiltration behavior)
        for rain_level in ['none', 'light', 'moderate', 'heavy']:
            for texture in ['sandy', 'loam', 'clayey']:
                mask = (df['rain_intensity'] == rain_level) & (
                    df['texture_class'] == texture)
                sub = df[mask]
                if len(sub) > 50:
                    interactions.append({
                        'interaction': 'rain_x_texture',
                        'condition1': rain_level,
                        'condition2': texture,
                        'n_samples': len(sub),
                        'mean_residual': sub['residual'].mean(),
                        'std_residual': sub['residual'].std(),
                        'interpretation': self._interpret_rain_texture(
                            rain_level, texture, sub['residual'].mean()
                        )
                    })

        # ET × Physics State interaction (ET stress behavior)
        for et_level in ['low', 'moderate', 'high', 'very_high']:
            for state in ['dry', 'normal', 'wet']:
                if state == 'dry':
                    mask = (df['et_intensity'] == et_level) & (
                        df['physics_dry'] == 1)
                elif state == 'wet':
                    mask = (df['et_intensity'] == et_level) & (
                        df['physics_wet'] == 1)
                else:
                    mask = (df['et_intensity'] == et_level) & (
                        df['physics_dry'] == 0) & (df['physics_wet'] == 0)

                sub = df[mask]
                if len(sub) > 50:
                    interactions.append({
                        'interaction': 'et_x_moisture_state',
                        'condition1': et_level,
                        'condition2': state,
                        'n_samples': len(sub),
                        'mean_residual': sub['residual'].mean(),
                        'std_residual': sub['residual'].std(),
                        'interpretation': self._interpret_et_state(
                            et_level, state, sub['residual'].mean()
                        )
                    })

        return pd.DataFrame(interactions)

    def _interpret_rain_texture(self, rain: str, texture: str, residual: float) -> str:
        """Generate interpretation for rain-texture interaction."""
        if residual > 0.02:  # Physics too dry
            if rain in ['moderate', 'heavy'] and texture == 'sandy':
                return "Ksat too HIGH in sandy soils - water draining too fast"
            elif rain in ['moderate', 'heavy'] and texture == 'clayey':
                return "Infiltration too LOW in clay - runoff overestimated"
            elif rain == 'none':
                return "Drainage too FAST - soil drying faster than observed"
        elif residual < -0.02:  # Physics too wet
            if rain in ['moderate', 'heavy'] and texture == 'sandy':
                return "Ksat too LOW - water not infiltrating fast enough"
            elif rain == 'none':
                return "ET too LOW - soil staying wetter than observed"
        return "Residuals within acceptable range"

    def _interpret_et_state(self, et: str, state: str, residual: float) -> str:
        """Generate interpretation for ET-moisture state interaction."""
        if residual > 0.02:  # Physics too dry
            if et in ['high', 'very_high'] and state == 'dry':
                return "Missing ET STRESS - plants still transpiring when soil dry"
            elif et in ['low', 'moderate']:
                return "ET coefficient (Kc/Kcb) too HIGH"
        elif residual < -0.02:  # Physics too wet
            if et in ['high', 'very_high']:
                return "ET too LOW - not removing enough water"
            elif state == 'wet':
                return "Drainage too SLOW - soil staying saturated"
        return "Residuals within acceptable range"


# =============================================================================
# SITE CLUSTERING
# =============================================================================

class SiteClusterer:
    """
    Cluster sites by soil-climate regime for group-wise calibration.
    """

    def __init__(self, df: pd.DataFrame, n_clusters: int = 5):
        """
        Initialize clusterer.

        Args:
            df: Canonical table with site characteristics
            n_clusters: Number of clusters
        """
        self.df = df
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.site_features = None

    def fit(self) -> pd.DataFrame:
        """
        Cluster sites and return cluster assignments.

        Returns:
            DataFrame with site-level cluster assignments and profiles
        """
        # Aggregate to site level
        site_agg = self.df.groupby('station_id').agg({
            'clay_pct': 'first',
            'sand_pct': 'first',
            'organic_carbon_pct': 'first',
            'saturation': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'elevation_m': 'first',
            'precipitation_mm': 'mean',
            'et0_mm': 'mean',
            'temperature_mean_c': 'mean',
            'residual': ['mean', 'std'],
            'soil_moisture': ['mean', 'std'],
            'physics_prior': ['mean', 'std'],
        }).reset_index()

        # Flatten column names
        site_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in site_agg.columns]

        # Select clustering features
        cluster_features = [
            'clay_pct', 'sand_pct',
            'precipitation_mm_mean', 'et0_mm_mean',
            'residual_mean',
        ]

        # Add if available
        for col in ['latitude', 'elevation_m', 'organic_carbon_pct']:
            if col in site_agg.columns:
                cluster_features.append(col)

        # Filter to available and fill NaN
        available_features = [
            f for f in cluster_features if f in site_agg.columns]
        X = site_agg[available_features].fillna(
            site_agg[available_features].median())

        # Scale and cluster
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=42, n_init=10)
        site_agg['cluster'] = self.kmeans.fit_predict(X_scaled)

        self.site_features = site_agg
        return site_agg

    def get_cluster_profiles(self) -> List[ClusterProfile]:
        """Generate detailed profiles for each cluster."""
        if self.site_features is None:
            self.fit()

        profiles = []

        for cluster_id in range(self.n_clusters):
            cluster_sites = self.site_features[self.site_features['cluster'] == cluster_id]
            cluster_data = self.df[self.df['station_id'].isin(
                cluster_sites['station_id'])]

            # Determine dominant climate zone
            if cluster_data['precipitation_mm'].mean() > 5:
                climate = 'wet_tropical'
            elif cluster_data['precipitation_mm'].mean() < 2:
                climate = 'semi_arid'
            else:
                climate = 'savanna'

            profile = ClusterProfile(
                cluster_id=cluster_id,
                n_sites=len(cluster_sites),
                n_samples=len(cluster_data),
                mean_clay=cluster_data['clay_pct'].mean(),
                mean_sand=cluster_data['sand_pct'].mean(),
                mean_precip=cluster_data['precipitation_mm'].mean(),
                mean_et0=cluster_data['et0_mm'].mean(),
                climate_zone=climate,
                mean_residual=cluster_data['residual'].mean(),
                std_residual=cluster_data['residual'].std(),
                residual_skew=cluster_data['residual'].skew(),
                corrections=[]  # Will be filled by derive_corrections
            )
            profiles.append(profile)

        return profiles


# =============================================================================
# PARAMETER CORRECTION DERIVATION
# =============================================================================

class ParameterCorrector:
    """
    Derive physics parameter corrections from residual analysis.
    """

    # Physical parameter bounds
    PARAM_BOUNDS = {
        'ksat_multiplier': (0.2, 5.0),
        'porosity_adjustment': (-0.10, 0.10),
        'fc_adjustment': (-0.10, 0.10),
        'kcb_multiplier': (0.5, 1.5),
        'drainage_rate_multiplier': (0.3, 3.0),
        'et_stress_threshold': (0.10, 0.40),
    }

    def __init__(self, diagnostics: ResidualDiagnostics, clusterer: SiteClusterer):
        """
        Initialize parameter corrector.

        Args:
            diagnostics: Residual diagnostics object
            clusterer: Site clusterer object
        """
        self.diagnostics = diagnostics
        self.clusterer = clusterer
        self.corrections = {}

    def derive_global_corrections(self) -> Dict[str, PhysicsCorrection]:
        """
        Derive global corrections applicable to all sites.
        """
        df = self.diagnostics.df

        corrections = {}

        # 1. Overall bias correction
        mean_residual = df['residual'].mean()
        if abs(mean_residual) > 0.01:
            corrections['bias'] = PhysicsCorrection(
                parameter_name='bias_correction_additive',
                correction_type='additive',
                correction_value=mean_residual * 0.8,  # Partial correction
                confidence=0.9 if abs(mean_residual) > 0.03 else 0.7,
                applicable_conditions={},
                rationale=f"Global bias of {mean_residual:.4f} detected. "
                f"Physics is {'too dry' if mean_residual > 0 else 'too wet'}."
            )

        # 2. ET stress correction
        # If residuals are positive when physics is dry AND ET is high,
        # we need to add/strengthen ET stress function
        mask_stress = (df['physics_dry'] == 1) & (
            df['et_intensity'].isin(['high', 'very_high']))
        stress_residual = df[mask_stress]['residual'].mean(
        ) if mask_stress.sum() > 100 else 0

        if stress_residual > 0.02:
            corrections['et_stress'] = PhysicsCorrection(
                parameter_name='et_stress_factor',
                correction_type='multiplicative',
                correction_value=max(0.5, 1.0 - stress_residual * 2),
                confidence=0.8,
                applicable_conditions={'physics_prior': (0.0, 0.15)},
                rationale=f"ET stress underestimated. When soil is dry, physics predicts "
                f"{stress_residual:.4f} more ET than observed. "
                f"Recommend ET stress function: ET = ET₀ × f(θ)"
            )

        return corrections

    def derive_cluster_corrections(self, profiles: List[ClusterProfile]) -> Dict[int, List[PhysicsCorrection]]:
        """
        Derive cluster-specific corrections.
        """
        df = self.diagnostics.df
        cluster_corrections = {}

        for profile in profiles:
            corrections = []
            cluster_data = df[df['station_id'].isin(
                self.clusterer.site_features[
                    self.clusterer.site_features['cluster'] == profile.cluster_id
                ]['station_id']
            )]

            # 1. Ksat correction based on infiltration behavior
            rain_mask = cluster_data['precipitation_mm'] > 10
            if rain_mask.sum() > 50:
                rain_residual = cluster_data[rain_mask]['residual'].mean()

                if rain_residual > 0.03:  # Physics too dry after rain
                    # Water draining too fast OR not infiltrating
                    if profile.mean_sand > 60:
                        # Sandy soil - Ksat probably too high
                        correction = PhysicsCorrection(
                            parameter_name='ksat_multiplier',
                            correction_type='multiplicative',
                            correction_value=max(0.3, 1.0 - rain_residual * 3),
                            confidence=0.75,
                            applicable_conditions={
                                'sand_pct': (50, 100),
                                'precipitation_mm': (10, 100)
                            },
                            rationale=f"Cluster {profile.cluster_id} (sandy): After rain, "
                            f"physics is {rain_residual:.3f} too dry. "
                            f"Ksat too high - reduce by factor."
                        )
                        corrections.append(correction)
                    else:
                        # Clay soil - runoff probably overestimated
                        correction = PhysicsCorrection(
                            parameter_name='infiltration_efficiency',
                            correction_type='multiplicative',
                            correction_value=min(1.5, 1.0 + rain_residual * 2),
                            confidence=0.70,
                            applicable_conditions={
                                'clay_pct': (30, 100),
                                'precipitation_mm': (10, 100)
                            },
                            rationale=f"Cluster {profile.cluster_id} (clay): Runoff overestimated. "
                            f"Increase infiltration efficiency."
                        )
                        corrections.append(correction)

                elif rain_residual < -0.03:  # Physics too wet after rain
                    correction = PhysicsCorrection(
                        parameter_name='drainage_rate_multiplier',
                        correction_type='multiplicative',
                        correction_value=min(3.0, 1.0 - rain_residual * 3),
                        confidence=0.70,
                        applicable_conditions={
                            'precipitation_mm': (10, 100)
                        },
                        rationale=f"Cluster {profile.cluster_id}: After rain, "
                        f"physics is {-rain_residual:.3f} too wet. "
                        f"Drainage too slow."
                    )
                    corrections.append(correction)

            # 2. Porosity/storage correction based on persistent bias
            if abs(profile.mean_residual) > 0.03:
                if profile.mean_residual > 0:
                    # Persistent dry bias - increase storage
                    correction = PhysicsCorrection(
                        parameter_name='porosity_adjustment',
                        correction_type='additive',
                        correction_value=min(
                            0.08, profile.mean_residual * 0.5),
                        confidence=0.65,
                        applicable_conditions={},
                        rationale=f"Cluster {profile.cluster_id}: Persistent dry bias "
                        f"({profile.mean_residual:.3f}). "
                        f"Soil storage capacity likely underestimated."
                    )
                    corrections.append(correction)
                else:
                    # Persistent wet bias - decrease storage or increase drainage
                    correction = PhysicsCorrection(
                        parameter_name='porosity_adjustment',
                        correction_type='additive',
                        correction_value=max(-0.08,
                                             profile.mean_residual * 0.5),
                        confidence=0.65,
                        applicable_conditions={},
                        rationale=f"Cluster {profile.cluster_id}: Persistent wet bias "
                        f"({profile.mean_residual:.3f}). "
                        f"Soil storage capacity likely overestimated."
                    )
                    corrections.append(correction)

            profile.corrections = corrections
            cluster_corrections[profile.cluster_id] = corrections

        return cluster_corrections


# =============================================================================
# MAIN DIAGNOSTIC PIPELINE
# =============================================================================

def run_diagnostics(
    canonical_path: Path,
    output_dir: Path,
    n_clusters: int = 5
) -> Dict:
    """
    Run full diagnostic pipeline.

    Args:
        canonical_path: Path to canonical_table_train.csv
        output_dir: Output directory
        n_clusters: Number of site clusters

    Returns:
        Dictionary with all diagnostic results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ML-ASSISTED PHYSICS PARAMETER ESTIMATION")
    print("=" * 70)

    # Load data
    print("\n📊 Loading canonical table...")
    df = pd.read_csv(canonical_path)
    print(
        f"   Loaded {len(df):,} samples from {df['station_id'].nunique()} stations")

    # Initialize diagnostics
    print("\n🔍 Analyzing residuals...")
    diagnostics = ResidualDiagnostics(df)

    # Compute basic statistics
    stats = diagnostics.compute_residual_statistics()
    print("\n" + "=" * 70)
    print("RESIDUAL STATISTICS BY GROUP")
    print("=" * 70)
    print(stats.to_string(index=False))
    stats.to_csv(output_dir / "residual_statistics.csv", index=False)

    # Train explainer model
    print("\n🧠 Training residual explainer model...")
    model, importance = diagnostics.train_residual_explainer()
    print("\nTop features driving residuals:")
    print(importance.head(10).to_string(index=False))
    importance.to_csv(
        output_dir / "residual_feature_importance.csv", index=False)

    # Analyze interactions
    print("\n🔗 Analyzing physical interactions...")
    interactions = diagnostics.analyze_interactions()
    print("\nKey interactions:")
    print(interactions[['interaction', 'condition1', 'condition2',
                        'mean_residual', 'interpretation']].head(10).to_string(index=False))
    interactions.to_csv(output_dir / "residual_interactions.csv", index=False)

    # Cluster sites
    print(f"\n🗺️  Clustering sites into {n_clusters} soil-climate regimes...")
    clusterer = SiteClusterer(df, n_clusters=n_clusters)
    site_clusters = clusterer.fit()
    profiles = clusterer.get_cluster_profiles()

    print("\nCluster profiles:")
    for p in profiles:
        print(f"\n  Cluster {p.cluster_id} ({p.climate_zone}):")
        print(f"    Sites: {p.n_sites}, Samples: {p.n_samples:,}")
        print(f"    Clay: {p.mean_clay:.1f}%, Sand: {p.mean_sand:.1f}%")
        print(
            f"    Mean precip: {p.mean_precip:.1f} mm, Mean ET₀: {p.mean_et0:.1f} mm")
        print(
            f"    Mean residual: {p.mean_residual:+.4f} ({'dry bias' if p.mean_residual > 0 else 'wet bias'})")

    site_clusters.to_csv(output_dir / "site_clusters.csv", index=False)

    # Derive corrections
    print("\n⚙️  Deriving physics parameter corrections...")
    corrector = ParameterCorrector(diagnostics, clusterer)

    global_corrections = corrector.derive_global_corrections()
    cluster_corrections = corrector.derive_cluster_corrections(profiles)

    print("\n" + "=" * 70)
    print("RECOMMENDED PHYSICS CORRECTIONS")
    print("=" * 70)

    print("\n📍 GLOBAL CORRECTIONS:")
    for name, corr in global_corrections.items():
        print(f"\n  {corr.parameter_name}:")
        print(f"    Type: {corr.correction_type}")
        print(f"    Value: {corr.correction_value:.4f}")
        print(f"    Confidence: {corr.confidence:.0%}")
        print(f"    Rationale: {corr.rationale}")

    print("\n📍 CLUSTER-SPECIFIC CORRECTIONS:")
    for cluster_id, corrections in cluster_corrections.items():
        profile = profiles[cluster_id]
        print(
            f"\n  Cluster {cluster_id} ({profile.climate_zone}, n={profile.n_sites} sites):")
        if not corrections:
            print("    No significant corrections needed")
        for corr in corrections:
            print(f"\n    {corr.parameter_name}:")
            print(f"      Type: {corr.correction_type}")
            print(f"      Value: {corr.correction_value:.4f}")
            print(f"      Rationale: {corr.rationale}")

    # Save corrections as JSON-like config
    corrections_config = {
        'global': {name: {
            'parameter': c.parameter_name,
            'type': c.correction_type,
            'value': c.correction_value,
            'confidence': c.confidence,
            'rationale': c.rationale
        } for name, c in global_corrections.items()},
        'clusters': {
            str(cid): [{
                'parameter': c.parameter_name,
                'type': c.correction_type,
                'value': c.correction_value,
                'confidence': c.confidence,
                'conditions': c.applicable_conditions,
                'rationale': c.rationale
            } for c in corrections]
            for cid, corrections in cluster_corrections.items()
        }
    }

    import json
    with open(output_dir / "physics_corrections.json", 'w') as f:
        json.dump(corrections_config, f, indent=2)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 70)

    # Generate code recommendations
    print("\n🔧 Recommended code changes for adaptive_calibration.py:")

    if 'et_stress' in global_corrections:
        print("""
    # Add ET stress function
    def apply_et_stress(et_potential, soil_moisture, theta_wp, theta_fc):
        '''Reduce ET when soil is dry.'''
        stress_threshold = 0.15  # Derived from residual analysis
        if soil_moisture < stress_threshold:
            stress_factor = max(0.2, (soil_moisture - theta_wp) / (stress_threshold - theta_wp))
            return et_potential * stress_factor
        return et_potential
        """)

    if any(c.parameter_name == 'ksat_multiplier' for corrs in cluster_corrections.values() for c in corrs):
        print("""
    # Texture-dependent Ksat adjustment
    def adjust_ksat(ksat_base, sand_pct, clay_pct, climate_zone):
        '''Adjust Ksat based on residual analysis.'''
        if sand_pct > 60:
            # Sandy soils: reduce Ksat (water draining too fast)
            ksat_mult = 0.6
        elif clay_pct > 40:
            # Clay soils: may need macropore enhancement
            ksat_mult = 1.2
        else:
            ksat_mult = 1.0
        return ksat_base * ksat_mult
        """)

    print("\n" + "=" * 70)
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 70)

    return {
        'statistics': stats,
        'importance': importance,
        'interactions': interactions,
        'site_clusters': site_clusters,
        'profiles': profiles,
        'global_corrections': global_corrections,
        'cluster_corrections': cluster_corrections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose physics model from ML residuals"
    )
    parser.add_argument(
        "--canonical-table", type=Path,
        default=Path(
            "results/no_leakage_validation/canonical_table_train.csv"),
        help="Path to canonical table CSV"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/physics_diagnostics"),
        help="Output directory"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of site clusters"
    )

    args = parser.parse_args()

    run_diagnostics(
        canonical_path=args.canonical_table,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze validation results."""
import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

# Read validation results - use latest results
val = pd.read_csv('results/hydrology_fix_v1/validation_results.csv')
print('=== OVERALL VALIDATION RESULTS ===')
print(val.to_string(index=False))
print()

# Read per-site results
sites = pd.read_csv('results/hydrology_fix_v1/per_site_results_24h.csv')
print(f'=== PER-SITE SUMMARY (24h horizon) - {len(sites)} stations ===')
print(f'Split breakdown:')
print(sites['split'].value_counts())
print()

# Compute improvement per site
sites['improvement_pct'] = (
    sites['physics_rmse'] - sites['hybrid_rmse']) / sites['physics_rmse'] * 100

# Summary by split
print('=== PERFORMANCE BY SPLIT ===')
for split in sites['split'].unique():
    df = sites[sites['split'] == split]
    print(
        f'{split}: n={len(df)}, hybrid_rmse={df["hybrid_rmse"].mean():.4f}, physics_rmse={df["physics_rmse"].mean():.4f}, improvement={df["improvement_pct"].mean():.1f}%')
    print(
        f'  KGE: hybrid={df["hybrid_kge"].mean():.3f}, physics={df["physics_kge"].mean():.3f}')
print()

# Best and worst stations
print('=== TOP 5 BEST HYBRID PERFORMANCE (by RMSE) ===')
best = sites.nsmallest(5, 'hybrid_rmse')[
    ['station_id', 'region', 'hybrid_rmse', 'physics_rmse', 'improvement_pct', 'hybrid_kge']]
print(best.to_string(index=False))
print()

print('=== TOP 5 WORST HYBRID PERFORMANCE (by RMSE) ===')
worst = sites.nlargest(5, 'hybrid_rmse')[
    ['station_id', 'region', 'hybrid_rmse', 'physics_rmse', 'improvement_pct', 'hybrid_kge']]
print(worst.to_string(index=False))
print()

# Spatial test stations (unseen locations)
print('=== SPATIAL TEST STATIONS (unseen locations) ===')
spatial = sites[sites['split'] == 'test_spatial']
spatial_sorted = spatial.sort_values('improvement_pct', ascending=False)
print(spatial_sorted[['station_id', 'region', 'hybrid_rmse',
      'physics_rmse', 'improvement_pct', 'hybrid_kge']].to_string(index=False))

# Feature importance
print('\n=== TOP 20 FEATURE IMPORTANCE ===')
fi = pd.read_csv('results/hydrology_fix_v1/feature_importance.csv')
print(fi.head(20).to_string(index=False))

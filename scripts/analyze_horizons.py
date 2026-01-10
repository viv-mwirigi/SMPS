#!/usr/bin/env python3
"""Analyze validation results by forecast horizon."""
import pandas as pd
from pathlib import Path

results_dir = Path('results/ismn_validation')
csv_files = sorted(results_dir.glob('ismn_validation_results_*.csv'))

if not csv_files:
    print("No results found!")
    exit(1)

df = pd.read_csv(csv_files[-1])
df_clean = df[df['KGE'] > -5]

print('=' * 60)
print('HORIZON ANALYSIS (excluding outlier stations with KGE < -5)')
print('=' * 60)
print(f'Observations with KGE > -5: {len(df_clean)} / {len(df)}')
print()
print(f"{'Horizon':<12} {'RMSE':<10} {'KGE':<10} {'R²':<10}")
print('-' * 45)
print(
    f"{'0h (now)':<12} {df_clean['RMSE'].mean():.4f}     {df_clean['KGE'].mean():.4f}     {df_clean['R²'].mean():.4f}")

for h in ['24h', '72h', '168h']:
    rmse_col = f'RMSE_{h}'
    kge_col = f'KGE_{h}'
    r2_col = f'R2_{h}'
    if rmse_col in df_clean.columns:
        hdata = df_clean[df_clean[rmse_col].notna()]
        print(
            f"{h:<12} {hdata[rmse_col].mean():.4f}     {hdata[kge_col].mean():.4f}     {hdata[r2_col].mean():.4f}")

print()
print('=' * 60)
print('BY DEPTH (clean data)')
print('=' * 60)
for depth in sorted(df_clean['depth_cm'].unique()):
    d = df_clean[df_clean['depth_cm'] == depth]
    print(
        f"  {int(depth):>3} cm: RMSE={d['RMSE'].mean():.4f}, KGE={d['KGE'].mean():.3f}, n={len(d)}")

print()
print('=' * 60)
print('HORIZON DEGRADATION ANALYSIS')
print('=' * 60)
baseline_rmse = df_clean['RMSE'].mean()
baseline_kge = df_clean['KGE'].mean()

for h, days in [('24h', 1), ('72h', 3), ('168h', 7)]:
    rmse_col = f'RMSE_{h}'
    kge_col = f'KGE_{h}'
    if rmse_col in df_clean.columns:
        hdata = df_clean[df_clean[rmse_col].notna()]
        rmse_change = (hdata[rmse_col].mean() -
                       baseline_rmse) / baseline_rmse * 100
        kge_change = (hdata[kge_col].mean() -
                      baseline_kge) / abs(baseline_kge) * 100
        print(
            f"{h}: RMSE change = {rmse_change:+.2f}%, KGE change = {kge_change:+.2f}%")

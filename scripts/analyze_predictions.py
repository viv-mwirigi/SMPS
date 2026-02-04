#!/usr/bin/env python3
"""Analyze the matric potential validation predictions."""
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv(
        'results/matric_potential_v2_sequential/detailed_predictions_0h.csv')
    print('=== DETAILED PREDICTION STATISTICS (0h) ===')
    print(f'Rows: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')

    print('\n=== Key Statistics ===')
    for col in ['observed_volumetric', 'physics_prior_matric', 'residual_correction_matric', 'predicted_matric', 'predicted_volumetric']:
        if col in df.columns:
            print(
                f'{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}')

    print(
        f'\nCorrelation (obs_vol vs pred_vol): {df["observed_volumetric"].corr(df["predicted_volumetric"]):.3f}')
    rmse = np.sqrt(
        ((df["observed_volumetric"] - df["predicted_volumetric"])**2).mean())
    print(f'RMSE: {rmse:.4f}')

    # Error analysis
    df['error'] = df['predicted_volumetric'] - df['observed_volumetric']
    print('\n=== Biggest Errors (over-predict) ===')
    top5 = df.nlargest(5, 'error')
    print(top5[['station_id', 'date', 'observed_volumetric', 'predicted_volumetric',
          'physics_prior_matric', 'predicted_matric']].to_string())

    print('\n=== Biggest Errors (under-predict) ===')
    bottom5 = df.nsmallest(5, 'error')
    print(bottom5[['station_id', 'date', 'observed_volumetric',
          'predicted_volumetric', 'physics_prior_matric', 'predicted_matric']].to_string())

    # Per-site analysis
    print('\n=== Per-Site RMSE ===')
    site_metrics = df.groupby('station_id').apply(
        lambda x: pd.Series({
            'rmse': np.sqrt(((x['observed_volumetric'] - x['predicted_volumetric'])**2).mean()),
            'bias': (x['predicted_volumetric'] - x['observed_volumetric']).mean(),
            'n': len(x)
        })
    ).reset_index()
    site_metrics = site_metrics.sort_values('rmse', ascending=False)
    print(site_metrics.head(10).to_string())

    # Check if the issue is with specific sites
    print(f'\n=== Site RMSE Distribution ===')
    print(f'Median: {site_metrics["rmse"].median():.4f}')
    print(f'Mean: {site_metrics["rmse"].mean():.4f}')
    print(f'Std: {site_metrics["rmse"].std():.4f}')

    # Check relationship between observed and physics prior
    from swpps.physics.van_genuchten import theta_to_psi_tropical
    obs_psi = df['observed_volumetric'].apply(
        lambda x: theta_to_psi_tropical(x, 0.05))
    print(f'\n=== Observed Theta -> Psi Conversion ===')
    print(f'Observed theta mean: {df["observed_volumetric"].mean():.3f}')
    print(f'Observed psi mean: {obs_psi.mean():.3f} kPa')
    print(
        f'Physics prior psi mean: {df["physics_prior_matric"].mean():.3f} kPa')
    print(f'Predicted psi mean: {df["predicted_matric"].mean():.3f} kPa')


if __name__ == '__main__':
    main()

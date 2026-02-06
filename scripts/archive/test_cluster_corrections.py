#!/usr/bin/env python
"""Quick test for cluster-specific corrections."""
from smps.physics.adaptive_calibration import AdaptivePhysicsCalibrator, SiteCharacteristics

test_cases = [
    ('Cluster 0 - Semi-arid', 10.0, 35.0, 24, 60, 200, 700),
    ('Cluster 1 - Highland low clay', 0.0, 35.0, 17, 55, 1500, 1100),
    ('Cluster 2 - Moderate clay', 7.0, 35.0, 31, 45, 1100, 1350),
    ('Cluster 3 - HIGH clay E.Africa', -2.0, 36.0, 57, 25, 1500, 1800),
    ('Cluster 4 - Wet tropical', -2.0, 29.0, 40, 46, 2000, 4400),
]

print('CLUSTER-SPECIFIC CORRECTIONS TEST')
print('=' * 70)

for name, lat, lon, clay, sand, elev, precip in test_cases:
    site = SiteCharacteristics.estimate_from_location(
        latitude=lat, longitude=lon, sand_percent=sand, clay_percent=clay,
        elevation_m=elev, annual_precip_mm=precip,
    )
    site.elevation_m = elev
    site.mean_annual_precip_mm = precip

    calibrator = AdaptivePhysicsCalibrator(site)
    p = calibrator.params
    cluster = calibrator._estimate_cluster(site, precip/365)

    print(f'{name}:')
    print(
        f'  cluster={cluster}, ksat={p.ksat_multiplier:.2f}, drainage={p.drainage_coefficient:.2f}')
    print(
        f'  theta_s_adj={p.theta_s_adjustment:+.3f}, infil_eff={p.infiltration_efficiency:.2f}')

print('=' * 70)
print('DONE')

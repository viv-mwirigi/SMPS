#!/usr/bin/env python
"""Test the ML-assisted calibration functions."""

import json
from src.smps.physics.adaptive_calibration import (
    apply_et_stress_function,
    AdaptivePhysicsCalibrator,
    SiteCharacteristics,
)
import sys
sys.path.insert(0, '/home/viv/SMPS')


print("=" * 60)
print("TEST: ML-ASSISTED PHYSICS CALIBRATION")
print("=" * 60)

# Test 1: ET stress function
print("\n📊 Test 1: ET stress function")
et = apply_et_stress_function(5.0, 0.25)  # Above threshold
print(f"  θ=0.25 (above threshold): ET = {et:.2f} mm (expected 5.0)")
assert abs(et - 5.0) < 0.01, "ET should be unchanged above threshold"

et = apply_et_stress_function(5.0, 0.10)  # In stress zone
print(f"  θ=0.10 (stress zone): ET = {et:.2f} mm (expected ~1.43)")
assert 1.0 < et < 2.0, "ET should be reduced in stress zone"

et = apply_et_stress_function(5.0, 0.08)  # At wilting point
print(f"  θ=0.08 (wilting point): ET = {et:.2f} mm (expected 1.0)")
assert abs(et - 1.0) < 0.01, "ET should be at minimum at wilting point"

print("  ✓ ET stress function works correctly")

# Test 2: Loading and applying ML corrections
print("\n📊 Test 2: ML correction application")
site = SiteCharacteristics.estimate_from_location(
    latitude=7.0, longitude=35.0, sand_percent=50.0, clay_percent=30.0
)
calibrator = AdaptivePhysicsCalibrator(site)

ksat_before = calibrator.params.ksat_multiplier
infil_before = calibrator.params.infiltration_efficiency
print(f"  Before ML corrections:")
print(f"    ksat_multiplier = {ksat_before:.2f}")
print(f"    infiltration_efficiency = {infil_before:.2f}")

# Load and apply corrections
with open('results/physics_diagnostics/physics_corrections.json', 'r') as f:
    corrections = json.load(f)

calibrator.apply_ml_derived_corrections(corrections, cluster_id=2)

print(f"  After ML corrections (cluster 2 - savanna):")
print(f"    ksat_multiplier = {calibrator.params.ksat_multiplier:.2f}")
print(
    f"    infiltration_efficiency = {calibrator.params.infiltration_efficiency:.2f}")
print(f"    theta_s_adjustment = {calibrator.params.theta_s_adjustment:.4f}")

# Verify changes were applied
assert calibrator.params.ksat_multiplier != ksat_before, "Ksat should change"
print("  ✓ ML corrections applied successfully")

# Test 3: Cluster-specific corrections
print("\n📊 Test 3: Cluster profiles")
for cluster_id in range(5):
    cluster_corrs = corrections["clusters"].get(str(cluster_id), [])
    if cluster_corrs:
        params = [c["parameter"] for c in cluster_corrs]
        print(f"  Cluster {cluster_id}: {', '.join(params)}")
    else:
        print(f"  Cluster {cluster_id}: No corrections needed")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)

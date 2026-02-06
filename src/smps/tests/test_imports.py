"""
SWPPS Package Import Test.

Run this to verify all modules can be imported correctly.
"""

import sys


def test_imports():
    """Test all SWPPS module imports."""
    errors = []

    # Core modules
    modules = [
        ("swpps.core.types", "Core types"),
        ("swpps.core.config", "Core config"),
        ("swpps.core.constants", "Core constants"),
        ("swpps.core.exceptions", "Core exceptions"),
        ("swpps.physics.van_genuchten", "Van Genuchten physics"),
        ("swpps.physics.water_balance", "Water balance model"),
        ("swpps.data.weather", "Weather data"),
        ("swpps.data.sensors", "Sensor data"),
        ("swpps.data.quality", "Quality control"),
        ("swpps.features.engineering", "Feature engineering"),
        ("swpps.ml.hybrid_model", "Hybrid ML model"),
        ("swpps.prediction.forecaster", "Forecaster"),
        ("swpps.prediction.decision", "Decision engine"),
        ("swpps.actuation.irrigation", "Irrigation actuation"),
        ("swpps.validation.metrics", "Validation metrics"),
        ("swpps.validation.report", "Validation report"),
        ("swpps.validation.plotting", "Plotting utilities"),
        ("swpps.calibration.calibrate", "Calibration"),
        ("swpps.calibration.objective", "Calibration objectives"),
        ("swpps.pipeline", "Main pipeline"),
        ("swpps.utils", "Utilities"),
    ]

    print("Testing SWPPS imports...")
    print("=" * 60)

    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description}: {module_name}")
        except ImportError as e:
            errors.append((module_name, str(e)))
            print(f"✗ {description}: {module_name}")
            print(f"  Error: {e}")

    print("=" * 60)

    # Test main package import
    try:
        import smps
        print(f"\n✓ Main package imported: swpps v{swpps.__version__}")
    except ImportError as e:
        errors.append(("swpps", str(e)))
        print(f"\n✗ Main package import failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} import errors")
        for module, error in errors:
            print(f"  - {module}: {error}")
        return 1
    else:
        print("SUCCESS: All modules imported correctly")
        return 0


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    print("=" * 60)

    try:
        from smps.core.types import VanGenuchtenParams, SoilMoistureStatus

        # Test VG params
        vg = VanGenuchtenParams(
            theta_r=0.05,
            theta_s=0.45,
            alpha=0.05,
            n=1.5,
            K_sat=100.0,
        )

        # Test theta from psi
        theta = vg.theta_from_psi(-33)
        print(f"✓ VG theta(-33 kPa) = {theta:.3f} m³/m³")

        # Test psi from theta
        psi = vg.psi_from_theta(0.30)
        print(f"✓ VG psi(0.30) = {psi:.1f} kPa")

        # Test status from potential
        status = SoilMoistureStatus.from_potential(-50)
        print(f"✓ Status at -50 kPa: {status.name}")

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return 1

    try:
        from smps.physics.van_genuchten import estimate_van_genuchten_params

        # Test PTF
        params = estimate_van_genuchten_params(
            sand_percent=40,
            clay_percent=25,
            organic_matter_percent=2.0,
        )
        print(
            f"✓ PTF estimation: θs={params.theta_s:.2f}, α={params.alpha:.3f}, n={params.n:.2f}")

    except Exception as e:
        print(f"✗ PTF test failed: {e}")
        return 1

    try:
        from smps.validation.metrics import compute_metrics
        import numpy as np

        # Test metrics
        obs = np.array([-30, -50, -80, -100, -120])
        pred = np.array([-35, -48, -75, -105, -115])

        metrics = compute_metrics(obs, pred)
        print(
            f"✓ Metrics: RMSE={metrics.rmse:.1f} kPa, R²={metrics.r_squared:.3f}")

    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return 1

    print("=" * 60)
    print("SUCCESS: Basic functionality tests passed")
    return 0


if __name__ == "__main__":
    ret = test_imports()
    if ret == 0:
        ret = test_basic_functionality()
    sys.exit(ret)

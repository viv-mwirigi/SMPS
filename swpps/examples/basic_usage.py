"""
Example: Basic usage of the SWPPS (Soil Water Potential Prediction System)

This example demonstrates:
1. Creating a pipeline for a field plot
2. Running a prediction cycle
3. Interpreting the results
4. Making irrigation decisions

The key innovation of SWPPS is using matric potential (kPa) instead of
volumetric water content. This means:
- NO soil-specific calibration needed
- Universal thresholds work for ALL soil types
- Direct relationship to plant water stress
"""

from swpps import (
    create_pipeline,
    PipelineConfig,
    CROP_THRESHOLDS,
    IRRIGATION_THRESHOLDS,
    SoilMoistureStatus,
    MatricPotential,
)
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import SWPPS components


def main():
    """Demonstrate basic SWPPS usage."""

    print("=" * 60)
    print("SWPPS - Soil Water Potential Prediction System")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Show available crop thresholds
    # -------------------------------------------------------------------------
    print("\n1. Available Crop Thresholds (in kPa):")
    print("-" * 40)

    for crop, thresholds in list(CROP_THRESHOLDS.items())[:5]:
        print(f"  {crop:12s}: irrigate below {thresholds['irrigate_below_kpa']:5.0f} kPa, "
              f"refill to {thresholds['refill_to_kpa']:4.0f} kPa")
    print("  ...")
    print(f"  (Total: {len(CROP_THRESHOLDS)} crops defined)")

    # -------------------------------------------------------------------------
    # 2. Demonstrate matric potential interpretation
    # -------------------------------------------------------------------------
    print("\n2. Matric Potential Interpretation:")
    print("-" * 40)

    test_potentials = [-10, -33, -50, -100, -200, -500, -1500]
    for psi in test_potentials:
        status = SoilMoistureStatus.from_potential(psi)
        print(f"  ψ = {psi:5.0f} kPa → {status.value}")

    # -------------------------------------------------------------------------
    # 3. Create a pipeline (without sensors - demo mode)
    # -------------------------------------------------------------------------
    print("\n3. Creating Pipeline for Demo Plot:")
    print("-" * 40)

    # Create configuration
    config = PipelineConfig(
        site_id="demo_plot",
        latitude=1.2921,      # Example: Nairobi, Kenya
        longitude=36.8219,
        soil_texture="loam",
        crop_type="tomato",
        root_depth_m=0.30,
        device_id="",         # No device - demo mode
        actuation_enabled=False,
        prediction_enabled=True,
        training_enabled=False,
    )

    print(f"  Site ID: {config.site_id}")
    print(f"  Location: {config.latitude}°N, {config.longitude}°E")
    print(f"  Crop: {config.crop_type}")
    print(f"  Soil: {config.soil_texture}")

    # Create pipeline
    from swpps.pipeline import SWPPSPipeline
    pipeline = SWPPSPipeline(config)

    print(f"  Van Genuchten params: θs={pipeline.vg_params.theta_s:.3f}, "
          f"α={pipeline.vg_params.alpha:.4f}, n={pipeline.vg_params.n:.2f}")

    # -------------------------------------------------------------------------
    # 4. Run a prediction cycle
    # -------------------------------------------------------------------------
    print("\n4. Running Prediction Cycle:")
    print("-" * 40)

    result = pipeline.run_prediction_cycle()

    if result["success"]:
        print(f"  Current state: ψ = {result['current_state']['psi_kpa']:.0f} kPa "
              f"({result['current_state']['status']})")

        print("\n  Forecasts:")
        for horizon, pred in result["forecasts"].items():
            print(f"    +{horizon:3d}h: {pred['prediction_kpa']:6.1f} kPa "
                  f"(±{pred['uncertainty_kpa']:.1f})")

        print("\n  Decision:")
        decision = result["decision"]
        print(f"    Action: {decision['action']}")
        print(f"    Should irrigate: {decision['should_irrigate']}")
        if decision["amount_mm"] > 0:
            print(f"    Amount: {decision['amount_mm']:.1f} mm")
        print(f"    Reason: {decision['reason']}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")

    # -------------------------------------------------------------------------
    # 5. Demonstrate threshold universality
    # -------------------------------------------------------------------------
    print("\n5. Why Matric Potential is Universal:")
    print("-" * 40)
    print("""
    Traditional VWC approach:
      - Sandy soil at 15% VWC → DRY (plant stressed)
      - Clay soil at 15% VWC → WET (plant fine)
      - Need soil-specific calibration!

    SWPPS Matric Potential approach:
      - Sandy soil at -100 kPa → OPTIMAL (same energy state)
      - Clay soil at -100 kPa → OPTIMAL (same energy state)
      - Universal thresholds work everywhere!

    Key thresholds for ALL soils:
      - Field capacity:  -10 to -33 kPa
      - Optimal range:   -33 to -100 kPa
      - Stress onset:    -100 to -200 kPa
      - Wilting point:   -1500 kPa
    """)

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

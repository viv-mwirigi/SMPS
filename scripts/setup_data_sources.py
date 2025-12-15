#!/usr/bin/env python
"""
SMPS Data Sources Setup and Verification Script.

This script helps:
1. Verify all data source configurations
2. Test API connections
3. Fetch sample data from each source
4. Troubleshoot common issues

Usage:
    python scripts/setup_data_sources.py
    python scripts/setup_data_sources.py --test-all
    python scripts/setup_data_sources.py --test-isda
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import argparse
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(name: str, status: bool, message: str = ""):
    """Print a status line."""
    icon = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{icon}{reset} {name}: {message}")


def check_environment():
    """Check environment variables."""
    print_header("Environment Configuration")

    env_vars = {
        "ISDA_USERNAME": ("iSDA Africa", True),
        "ISDA_PASSWORD": ("iSDA Africa", True),
        "GEE_SERVICE_ACCOUNT": ("Google Earth Engine", False),
        "GEE_SERVICE_ACCOUNT_KEY": ("Google Earth Engine", False),
        "ISMN_USERNAME": ("ISMN Validation", False),
        "ISMN_PASSWORD": ("ISMN Validation", False),
    }

    configured = 0
    for var, (service, required) in env_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}":
            print_status(var, True, f"Configured for {service}")
            configured += 1
        else:
            status = "Required" if required else "Optional"
            print_status(var, not required, f"{status} - {service}")

    print(f"\n  {configured}/{len(env_vars)} environment variables configured")
    return configured >= 2  # At minimum, need iSDA credentials


def test_isda_connection():
    """Test iSDA Africa API connection."""
    print_header("Testing iSDA Africa API")

    from smps.data.sources.isda_authenticated import IsdaAfricaAuthenticatedSource

    source = IsdaAfricaAuthenticatedSource()

    # Check credentials
    if not source.username or not source.password:
        print_status("Credentials", False, "Not configured in .env")
        print("\n  To configure:")
        print("  1. Register at https://www.isda-africa.com/isdasoil/")
        print("  2. Add ISDA_USERNAME and ISDA_PASSWORD to .env")
        return False

    print_status("Credentials", True, f"Using {source.username}")

    # Test authentication
    try:
        if source._authenticate():
            print_status("Authentication", True, "Token obtained successfully")
        else:
            print_status("Authentication", False, "Login failed")
            return False
    except Exception as e:
        print_status("Authentication", False, str(e))
        return False

    # Test data fetch
    try:
        # Test location in Kenya
        lat, lon = -0.7196, 35.2400
        profile = source.fetch_soil_profile(
            "test_kenya",
            latitude=lat,
            longitude=lon,
            depth="0-20"
        )

        print_status("Data Fetch", True, f"Location: ({lat}, {lon})")
        print(f"\n  Sample data retrieved:")
        print(f"    Sand: {profile.sand_percent}%")
        print(f"    Clay: {profile.clay_percent}%")
        print(f"    Porosity: {profile.porosity}")
        print(f"    Field Capacity: {profile.field_capacity}")
        return True

    except Exception as e:
        print_status("Data Fetch", False, str(e))
        return False


def test_soilgrids_connection():
    """Test SoilGrids API connection."""
    print_header("Testing SoilGrids API")

    from smps.data.sources.soilgrids import SoilGridsGlobalSource

    source = SoilGridsGlobalSource()
    print_status("Configuration", True, "No authentication required")

    # Test data fetch - use a location outside Africa
    try:
        lat, lon = 52.1, 5.2  # Netherlands
        profile = source.fetch_soil_profile(
            "test_netherlands",
            latitude=lat,
            longitude=lon,
            depth="0-5cm"
        )

        print_status("Data Fetch", True, f"Location: ({lat}, {lon})")
        print(f"\n  Sample data retrieved:")
        print(f"    Sand: {profile.sand_percent}%")
        print(f"    Clay: {profile.clay_percent}%")
        print(f"    Source: {profile.source}")
        return True

    except Exception as e:
        print_status("Data Fetch", False, str(e))
        return False


def test_gee_connection():
    """Test Google Earth Engine connection."""
    print_header("Testing Google Earth Engine")

    try:
        from smps.data.sources.gee_satellite import GoogleEarthEngineSatelliteSource
    except ImportError as e:
        print_status("Import", False, f"Missing dependency: {e}")
        print("\n  Install with: pip install earthengine-api")
        return False

    source = GoogleEarthEngineSatelliteSource()

    # Check credentials
    if not source.config.is_configured:
        print_status("Configuration", False, "Service account not configured")
        print("\n  To configure GEE:")
        print("  1. Create Google Cloud project")
        print("  2. Enable Earth Engine API")
        print("  3. Create service account with EE access")
        print("  4. Download JSON key and set GEE_SERVICE_ACCOUNT_KEY")

        # Try interactive authentication
        print("\n  Alternatively, run: earthengine authenticate")

        # Still test with synthetic data
        print("\n  Testing with synthetic data fallback...")
    else:
        print_status("Configuration", True, "Service account configured")

    # Test initialization
    if source.initialize():
        print_status("Initialization", True, "GEE initialized")
    else:
        print_status("Initialization", False, "Using synthetic data fallback")

    # Test data fetch (will use synthetic if GEE unavailable)
    try:
        lat, lon = -1.2921, 36.8219  # Nairobi
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        ndvi_data = source.fetch_ndvi(lat, lon, start_date, end_date)

        if ndvi_data:
            print_status("NDVI Fetch", True,
                         f"Got {len(ndvi_data)} observations")
            latest = ndvi_data[-1]
            print(f"\n  Latest NDVI: {latest.ndvi:.4f}")
            print(f"  Source: {latest.source}")
        return True

    except Exception as e:
        print_status("Data Fetch", False, str(e))
        return False


def test_weather_connection():
    """Test Open-Meteo weather API."""
    print_header("Testing Open-Meteo Weather API")

    from smps.data.sources.weather import OpenMeteoSource
    from smps.data.sources.base import DataFetchRequest

    source = OpenMeteoSource()
    print_status("Configuration", True, "No authentication required")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        request = DataFetchRequest(
            site_id="test_nairobi",
            start_date=start_date.date(),
            end_date=end_date.date()
        )

        weather = source.fetch_daily_weather(request)

        print_status("Data Fetch", True, f"Got {len(weather)} days")

        if weather:
            latest = weather[-1]
            print(f"\n  Latest date: {latest.date}")
            print(
                f"  Temp: {latest.temperature_min_c:.1f}-{latest.temperature_max_c:.1f}°C")
            print(f"  Precip: {latest.precipitation_mm:.1f}mm")
            print(f"  ET₀: {latest.et0_mm:.2f}mm/day")
        return True

    except Exception as e:
        print_status("Data Fetch", False, str(e))
        return False


def test_all_sources():
    """Test all data sources."""
    results = {}

    print("\n" + "=" * 60)
    print(" SMPS Data Sources Verification")
    print("=" * 60)

    # Check environment first
    results["environment"] = check_environment()

    # Test each source
    results["isda"] = test_isda_connection()
    results["soilgrids"] = test_soilgrids_connection()
    results["gee"] = test_gee_connection()
    results["weather"] = test_weather_connection()

    # Summary
    print_header("Summary")

    passed = sum(results.values())
    total = len(results)

    for name, status in results.items():
        print_status(name.upper(), status, "OK" if status else "FAILED")

    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        print("\n  See above for configuration instructions.")

    return passed == total


def print_setup_guide():
    """Print complete setup guide."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         SMPS Data Sources Setup Guide                        ║
╚══════════════════════════════════════════════════════════════╝

1. iSDA AFRICA (African Soil Data - 30m resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Register: https://www.isda-africa.com/isdasoil/
   Add to .env:
     ISDA_USERNAME=your_email@example.com
     ISDA_PASSWORD=your_password

2. SOILGRIDS (Global Soil Data - 250m resolution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   No registration required! Free REST API.
   Docs: https://rest.isric.org/soilgrids/v2.0/docs

3. GOOGLE EARTH ENGINE (Satellite Data - NDVI, LAI, LST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   a) Sign up: https://earthengine.google.com/signup/
   b) Create GCP project: https://console.cloud.google.com/
   c) Enable Earth Engine API
   d) Create service account with "Earth Engine Resource Writer" role
   e) Download JSON key
   f) Add to .env:
        GEE_SERVICE_ACCOUNT=account@project.iam.gserviceaccount.com
        GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json
        GEE_PROJECT_ID=your-project-id

   Alternative (interactive):
     pip install earthengine-api
     earthengine authenticate

4. OPEN-METEO (Weather Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   No registration required! Free API (10k requests/day).
   Docs: https://open-meteo.com/en/docs

5. ISMN (Validation Data - optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Register: https://ismn.geo.tuwien.ac.at/en/
   Add to .env:
     ISMN_USERNAME=your_username
     ISMN_PASSWORD=your_password

REQUIRED PACKAGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   pip install requests numpy pandas python-dotenv
   pip install earthengine-api  # For GEE

QUICK START:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. cp .env.example .env
   2. Edit .env with your credentials
   3. python scripts/setup_data_sources.py --test-all
""")


def main():
    parser = argparse.ArgumentParser(description="SMPS Data Sources Setup")
    parser.add_argument("--test-all", action="store_true",
                        help="Test all data source connections")
    parser.add_argument("--test-isda", action="store_true",
                        help="Test iSDA Africa connection only")
    parser.add_argument("--test-gee", action="store_true",
                        help="Test Google Earth Engine only")
    parser.add_argument("--test-soilgrids", action="store_true",
                        help="Test SoilGrids only")
    parser.add_argument("--test-weather", action="store_true",
                        help="Test weather API only")
    parser.add_argument("--guide", action="store_true",
                        help="Print setup guide")

    args = parser.parse_args()

    if args.guide:
        print_setup_guide()
    elif args.test_isda:
        test_isda_connection()
    elif args.test_gee:
        test_gee_connection()
    elif args.test_soilgrids:
        test_soilgrids_connection()
    elif args.test_weather:
        test_weather_connection()
    elif args.test_all:
        test_all_sources()
    else:
        print_setup_guide()
        print("\nRun with --test-all to verify your configuration.")


if __name__ == "__main__":
    main()

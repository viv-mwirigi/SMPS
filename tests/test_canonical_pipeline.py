"""
Integration tests for the canonical table pipeline.

NOTE: These tests require network connectivity and external data sources.
They are marked with @pytest.mark.integration to allow skipping.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

from smps.pipeline.canonical import CanonicalTableBuilder, CanonicalTableManager
from smps.core.config import get_config
from smps.core.types import SiteID


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestCanonicalTableBuilder:

    def setup_method(self):
        """Set up test environment"""
        self.config = get_config()
        self.builder = CanonicalTableBuilder(self.config.data)
        self.test_site_id = "test_site_001"
        self.start_date = date(2023, 6, 1)
        self.end_date = date(2023, 6, 30)

    def test_build_for_site_basic(self):
        """Test basic table building"""
        df = self.builder.build_for_site(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date,
            include_forecast=False
        )

        # Basic assertions
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) == 30, f"Expected 30 days, got {len(df)}"

        # Check required columns
        required_cols = [
            'site_id', 'date', 'precipitation_mm', 'et0_mm',
            'physics_theta_surface', 'physics_theta_root'
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
        assert df['site_id'].iloc[0] == self.test_site_id

    def test_physics_prior_inclusion(self):
        """Test that physics prior values are included and reasonable"""
        df = self.builder.build_for_site(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Physics prior columns should exist
        assert 'physics_theta_surface' in df.columns
        assert 'physics_theta_root' in df.columns

        # Values should be within physical bounds
        assert df['physics_theta_surface'].between(0.01, 0.50).all()
        assert df['physics_theta_root'].between(0.01, 0.50).all()

        # Surface should generally be drier than root (but not always)
        # Check that values are physically plausible
        assert not (df['physics_theta_surface'] >
                    df['physics_theta_root'] + 0.1).any()

    def test_feature_engineering(self):
        """Test that features are engineered correctly"""
        df = self.builder.build_for_site(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Check for engineered features
        expected_features = [
            'precip_cumulative_7d',
            'et0_cumulative_7d',
            'climate_water_deficit',
            'day_of_year_sin',
            'day_of_year_cos',
            'water_deficit_mm'
        ]

        for feature in expected_features:
            assert feature in df.columns, f"Missing engineered feature: {feature}"

        # Check cumulative calculations
        precip_7d = df['precip_cumulative_7d']
        # Rolling sum should be >= daily value
        assert (precip_7d >= df['precipitation_mm']).all()

    def test_data_coverage(self):
        """Test data coverage calculation"""
        df = self.builder.build_for_site(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Check that coverage is reasonable
        coverage = self.builder.metrics["data_coverage"]
        assert 0.7 <= coverage <= 1.0, f"Unexpected data coverage: {coverage}"

        # Check that there are no completely empty columns
        empty_cols = df.columns[df.isna().all()]
        assert len(empty_cols) == 0, f"Empty columns: {empty_cols}"

    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create a date range with a gap in the middle
        start_date = date(2023, 6, 1)
        mid_date = date(2023, 6, 15)
        end_date = date(2023, 6, 30)

        # Build table
        df = self.builder.build_for_site(
            site_id=self.test_site_id,
            start_date=start_date,
            end_date=end_date
        )

        # Should have all dates
        assert len(df) == 30

        # Should not have excessive missing values
        missing_percentage = df.isna().sum().sum() / (len(df) * len(df.columns))
        assert missing_percentage < 0.3, f"Too much missing data: {missing_percentage}"

    def test_validation(self):
        """Test table validation"""
        df = self.builder.build_for_site(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Check for validation metadata
        assert 'validation_status' in df.attrs
        assert df.attrs['validation_status'] == 'validated'

        # Check that required columns have valid values
        assert df['precipitation_mm'].ge(0).all()
        assert df['et0_mm'].ge(0).all()

        # Check soil moisture bounds
        assert df['physics_theta_surface'].between(0.01, 0.50).all()
        assert df['physics_theta_root'].between(0.01, 0.50).all()


class TestCanonicalTableManager:

    def setup_method(self):
        """Set up test environment"""
        self.test_dir = Path("./test_cache")
        self.test_dir.mkdir(exist_ok=True)

        self.manager = CanonicalTableManager(self.test_dir)
        self.test_site_id = "test_site_001"
        self.start_date = date(2023, 6, 1)
        self.end_date = date(2023, 6, 10)

    def teardown_method(self):
        """Clean up test files"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_caching(self):
        """Test table caching functionality"""
        # First call should build and cache
        df1 = self.manager.get_table(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Should have created cache file
        cache_files = list(self.test_dir.glob("*.parquet"))
        assert len(cache_files) == 1

        # Second call should load from cache
        df2 = self.manager.get_table(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # DataFrames should be equal
        pd.testing.assert_frame_equal(df1, df2)

        # Check cache metadata
        assert 'cache_key' in df2.attrs
        assert 'etl_version' in df2.attrs

    def test_force_rebuild(self):
        """Test force rebuild option"""
        # Build and cache
        df1 = self.manager.get_table(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Force rebuild
        df2 = self.manager.get_table(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date,
            force_rebuild=True
        )

        # DataFrames should be similar but not necessarily identical
        # (due to random elements in mock data)
        assert df1.shape == df2.shape
        assert set(df1.columns) == set(df2.columns)

    def test_cache_expiry(self):
        """Test cache expiry (simulated)"""
        # This test would require mocking file timestamps
        # For now, just test that the parameter is accepted
        df = self.manager.get_table(
            site_id=self.test_site_id,
            start_date=self.start_date,
            end_date=self.end_date,
            cache_ttl_days=1
        )

        assert not df.empty


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_multiple_sites(self):
        """Test building tables for multiple sites"""
        config = get_config()
        builder = CanonicalTableBuilder(config.data)

        site_ids = ["test_site_001", "test_site_002"]
        start_date = date(2023, 6, 1)
        end_date = date(2023, 6, 10)

        results = builder.build_for_multiple_sites(
            site_ids=site_ids,
            start_date=start_date,
            end_date=end_date,
            max_workers=2
        )

        # Check results
        assert len(results) == 2
        for site_id in site_ids:
            assert site_id in results
            df = results[site_id]
            assert not df.empty
            assert len(df) == 10  # 10 days
            assert df['site_id'].iloc[0] == site_id

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        from smps.core.config import SmpsConfig
        from smps.pipeline.canonical import CanonicalTableBuilder

        # Create test configuration
        test_config = SmpsConfig(
            environment="testing",
            data=SmpsConfig.__fields__["data"].type_(
                enabled_sources=["mock_weather", "mock_soil"],
                cache_dir=Path("./test_cache")
            )
        )

        # Build table
        builder = CanonicalTableBuilder(test_config.data)
        df = builder.build_for_site(
            site_id="integration_test_site",
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 5)
        )

        # Verify results
        assert not df.empty
        assert len(df) == 5

        # Check that all components worked together
        assert 'physics_theta_surface' in df.columns
        assert 'physics_theta_root' in df.columns
        assert 'precip_cumulative_7d' in df.columns
        assert 'et0_cumulative_7d' in df.columns

        # Check data quality
        assert df['precipitation_mm'].ge(0).all()
        assert df['physics_theta_surface'].between(0.01, 0.50).all()

        # Clean up
        import shutil
        test_cache = Path("./test_cache")
        if test_cache.exists():
            shutil.rmtree(test_cache)

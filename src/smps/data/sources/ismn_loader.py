"""
ISMN (International Soil Moisture Network) Station Data Loader.

Reads ISMN data files in the standard format:
- *_sm_*_*.stm files for soil moisture timeseries
- *_static_variables.csv for soil properties

Features:
- Parses metadata (depth, lat/lon, sensor, station_id, soil properties)
- Cleans -9999 missing values
- Filters to quality flags G (good) or M (missing but usable)
- Aggregates to daily soil moisture (θ) per depth
"""
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger("smps.data.ismn_loader")


# Quality flags considered acceptable
ACCEPTABLE_QUALITY_FLAGS = {"G", "M"}
MISSING_VALUE = -9999


@dataclass
class ISMNSensorMetadata:
    """Metadata for a single ISMN sensor/depth combination."""
    network: str
    station: str
    latitude: float
    longitude: float
    elevation: float
    depth_from_m: float
    depth_to_m: float
    sensor_type: str
    start_date: date
    end_date: date
    file_path: Path

    @property
    def depth_cm(self) -> float:
        """Depth in centimeters (average of from/to)."""
        return (self.depth_from_m + self.depth_to_m) / 2 * 100

    @property
    def station_id(self) -> str:
        """Unique station identifier."""
        return f"{self.network}_{self.station}"

    @property
    def sensor_id(self) -> str:
        """Unique sensor identifier including depth."""
        return f"{self.station_id}_{self.sensor_type}_{self.depth_cm:.0f}cm"


@dataclass
class ISMNSoilProperties:
    """Soil properties from static variables file."""
    depth_from_m: float
    depth_to_m: float
    saturation: Optional[float] = None  # m³/m³
    clay_fraction: Optional[float] = None  # % weight
    sand_fraction: Optional[float] = None  # % weight
    silt_fraction: Optional[float] = None  # % weight
    organic_carbon: Optional[float] = None  # % weight


@dataclass
class ISMNStationData:
    """Complete data for an ISMN station."""
    network: str
    station: str
    latitude: float
    longitude: float
    elevation: float
    sensors: Dict[str, ISMNSensorMetadata] = field(default_factory=dict)
    soil_properties: Dict[str, ISMNSoilProperties] = field(
        default_factory=dict)
    land_cover: Optional[str] = None
    climate_classification: Optional[str] = None
    daily_data: Optional[pd.DataFrame] = None  # Aggregated daily θ by depth

    @property
    def station_id(self) -> str:
        """Unique station identifier."""
        return f"{self.network}_{self.station}"

    @property
    def available_depths_cm(self) -> List[float]:
        """List of available measurement depths in cm."""
        return sorted(set(s.depth_cm for s in self.sensors.values()))


class ISMNStationLoader:
    """
    Loader for ISMN station data from downloaded files.

    Handles the standard ISMN file format with:
    - Network/Station folder structure
    - *_sm_*_*.stm files for soil moisture
    - *_static_variables.csv for metadata
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            base_path: Base path to ISMN data directory. If None, will look
                      in default locations.
        """
        self.base_path = Path(
            base_path) if base_path else self._find_default_path()

    def _find_default_path(self) -> Path:
        """Find default ISMN data path."""
        possible_paths = [
            Path("./data/ismn"),
            Path("../data/ismn"),
            Path.home() / "data" / "ismn",
        ]
        for p in possible_paths:
            if p.exists():
                return p
        raise FileNotFoundError("Could not find ISMN data directory")

    def list_available_datasets(self) -> List[Path]:
        """List available ISMN datasets (downloaded data packages)."""
        if not self.base_path.exists():
            return []
        # Support passing either:
        # 1) a parent directory containing multiple ISMN datasets (Data_*/...)
        # 2) a single dataset directory itself (Data_*/...)
        if self.base_path.is_dir() and self.base_path.name.startswith("Data_"):
            return [self.base_path]

        return [
            d
            for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("Data_")
        ]

    def list_networks(self, dataset_path: Optional[Path] = None) -> List[str]:
        """List available networks in a dataset."""
        if dataset_path is None:
            datasets = self.list_available_datasets()
            if not datasets:
                return []
            dataset_path = datasets[0]

        networks = []
        for item in dataset_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                networks.append(item.name)
        return sorted(networks)

    def list_stations(self, network: str, dataset_path: Optional[Path] = None) -> List[str]:
        """List available stations in a network."""
        if dataset_path is None:
            datasets = self.list_available_datasets()
            if not datasets:
                return []
            dataset_path = datasets[0]

        network_path = dataset_path / network
        if not network_path.exists():
            return []

        stations = []
        for item in network_path.iterdir():
            if item.is_dir():
                stations.append(item.name)
        return sorted(stations)

    def load_station(
        self,
        station_folder: Path,
        filter_quality: bool = True,
        aggregate_daily: bool = True
    ) -> ISMNStationData:
        """
        Load all data for a single station.

        Args:
            station_folder: Path to the station folder containing .stm files
            filter_quality: If True, only keep records with G/M quality flags
            aggregate_daily: If True, aggregate to daily averages

        Returns:
            ISMNStationData object with all sensor data and metadata
        """
        station_folder = Path(station_folder)

        if not station_folder.exists():
            raise FileNotFoundError(
                f"Station folder not found: {station_folder}")

        # Find all soil moisture files
        sm_files = list(station_folder.glob("*_sm_*_*.stm"))

        if not sm_files:
            raise ValueError(
                f"No soil moisture files found in {station_folder}")

        # Load first file to get station metadata
        first_metadata = self._parse_stm_header(sm_files[0])

        # Initialize station data
        station_data = ISMNStationData(
            network=first_metadata.network,
            station=first_metadata.station,
            latitude=first_metadata.latitude,
            longitude=first_metadata.longitude,
            elevation=first_metadata.elevation
        )

        # Load static variables (soil properties)
        static_file = list(station_folder.glob("*_static_variables.csv"))
        if static_file:
            self._load_static_variables(station_data, static_file[0])

        # Load all sensor files
        all_timeseries = []
        for sm_file in sm_files:
            try:
                metadata, timeseries = self._load_stm_file(
                    sm_file, filter_quality=filter_quality
                )
                station_data.sensors[metadata.sensor_id] = metadata

                if not timeseries.empty:
                    timeseries["depth_cm"] = metadata.depth_cm
                    timeseries["sensor_type"] = metadata.sensor_type
                    all_timeseries.append(timeseries)

            except Exception as e:
                logger.warning(f"Failed to load {sm_file}: {e}")

        # Combine all timeseries
        if all_timeseries:
            combined = pd.concat(all_timeseries, ignore_index=True)

            if aggregate_daily:
                station_data.daily_data = self._aggregate_to_daily(combined)
            else:
                station_data.daily_data = combined

        return station_data

    def _parse_stm_header(self, file_path: Path) -> ISMNSensorMetadata:
        """Parse the header line of an STM file to extract metadata."""
        # Also extract metadata from filename
        filename = file_path.name
        filename_match = re.match(
            r"(.+?)_(.+?)_(.+?)_sm_(\d+\.\d+)_(\d+\.\d+)_(.+?)_(\d{8})_(\d{8})\.stm",
            filename
        )

        with open(file_path, "r") as f:
            header_line = f.readline().strip()

        # Parse header: CSE Network Station Lat Lon Elevation DepthFrom DepthTo Sensor
        # Example: AMMA-CATCH AMMA-CATCH Belefoungou-Mid 9.79506 1.70994 414.0 0.1000 0.1000 CS616
        parts = header_line.split()

        # Header may have sensor on second line due to formatting
        if len(parts) < 9:
            # Read more and try to combine
            second_line = f.readline().strip() if f else ""
            parts.extend(second_line.split())

        # Find the numeric values - they should be lat, lon, elevation, depth_from, depth_to
        numeric_indices = []
        for i, p in enumerate(parts):
            try:
                float(p)
                numeric_indices.append(i)
            except ValueError:
                pass

        # Extract values based on expected positions
        network = parts[0] if len(parts) > 0 else "Unknown"
        # Station is between network and first numeric
        if numeric_indices:
            station = "_".join(parts[2:numeric_indices[0]])
            lat = float(parts[numeric_indices[0]])
            lon = float(parts[numeric_indices[1]])
            elev = float(parts[numeric_indices[2]])
            depth_from = float(parts[numeric_indices[3]])
            depth_to = float(parts[numeric_indices[4]])
        else:
            # Fallback: use filename info
            station = filename_match.group(3) if filename_match else "Unknown"
            lat, lon, elev = 0.0, 0.0, 0.0
            depth_from = float(filename_match.group(4)
                               ) if filename_match else 0.0
            depth_to = float(filename_match.group(
                5)) if filename_match else 0.0

        # Sensor type is typically at the end or from filename
        if filename_match:
            sensor_type = filename_match.group(6)
            start_date = datetime.strptime(
                filename_match.group(7), "%Y%m%d").date()
            end_date = datetime.strptime(
                filename_match.group(8), "%Y%m%d").date()
        else:
            sensor_type = parts[-1] if parts else "Unknown"
            start_date = date(2000, 1, 1)
            end_date = date(2025, 1, 1)

        return ISMNSensorMetadata(
            network=network,
            station=station,
            latitude=lat,
            longitude=lon,
            elevation=elev,
            depth_from_m=depth_from,
            depth_to_m=depth_to,
            sensor_type=sensor_type,
            start_date=start_date,
            end_date=end_date,
            file_path=file_path
        )

    def _load_stm_file(
        self,
        file_path: Path,
        filter_quality: bool = True
    ) -> Tuple[ISMNSensorMetadata, pd.DataFrame]:
        """
        Load a single STM file.

        Args:
            file_path: Path to the .stm file
            filter_quality: If True, filter to G/M quality flags only

        Returns:
            Tuple of (metadata, timeseries DataFrame)
        """
        metadata = self._parse_stm_header(file_path)

        # Read data lines (skip header)
        records = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # Skip header

                line = line.strip()
                if not line:
                    continue

                try:
                    record = self._parse_data_line(line)
                    if record is not None:
                        records.append(record)
                except Exception as e:
                    logger.debug(
                        f"Failed to parse line {i} in {file_path}: {e}")

        if not records:
            return metadata, pd.DataFrame()

        df = pd.DataFrame(records)

        # Clean -9999 missing values
        df.loc[df["soil_moisture"] == MISSING_VALUE, "soil_moisture"] = np.nan

        # Filter quality flags
        if filter_quality:
            df = df[df["ismn_quality_flag"].isin(ACCEPTABLE_QUALITY_FLAGS)]

        # Remove rows with missing soil moisture
        df = df.dropna(subset=["soil_moisture"])

        return metadata, df

    def _parse_data_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single data line from STM file.

        Format: YYYY/MM/DD HH:MM value ismn_flag [provider_flag]
        Example: 2017/01/05 00:00 0.0597 G M
        """
        parts = line.split()

        if len(parts) < 4:
            return None

        # Parse datetime
        date_str = parts[0]
        time_str = parts[1]
        try:
            timestamp = datetime.strptime(
                f"{date_str} {time_str}", "%Y/%m/%d %H:%M")
        except ValueError:
            return None

        # Parse value
        try:
            value = float(parts[2])
        except ValueError:
            return None

        # Parse flags
        ismn_flag = parts[3] if len(parts) > 3 else "U"
        provider_flag = parts[4] if len(parts) > 4 else None

        return {
            "timestamp": timestamp,
            "soil_moisture": value,
            "ismn_quality_flag": ismn_flag,
            "provider_quality_flag": provider_flag
        }

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sub-daily measurements to daily averages per depth.

        Args:
            df: DataFrame with timestamp, soil_moisture, depth_cm columns

        Returns:
            DataFrame with date, depth_cm, and daily statistics
        """
        if df.empty:
            return pd.DataFrame()

        # Extract date
        df = df.copy()
        df["date"] = df["timestamp"].dt.date

        # Group by date and depth
        daily = df.groupby(["date", "depth_cm"]).agg(
            soil_moisture_mean=("soil_moisture", "mean"),
            soil_moisture_min=("soil_moisture", "min"),
            soil_moisture_max=("soil_moisture", "max"),
            soil_moisture_std=("soil_moisture", "std"),
            n_observations=("soil_moisture", "count"),
            sensor_type=("sensor_type", "first")
        ).reset_index()

        # Convert date back to datetime for consistency
        daily["date"] = pd.to_datetime(daily["date"])

        return daily

    def _load_static_variables(
        self,
        station_data: ISMNStationData,
        file_path: Path
    ) -> None:
        """Load static variables (soil properties) from CSV file."""
        try:
            df = pd.read_csv(file_path, sep=";")

            # Parse soil properties by depth
            soil_props = {}

            for _, row in df.iterrows():
                quantity = row.get("quantity_name", "")
                depth_from = row.get("depth_from[m]", 0)
                depth_to = row.get("depth_to[m]", 0)
                value = row.get("value", None)

                # Handle land cover and climate (no depth values)
                if quantity == "land cover classification":
                    if pd.notna(row.get("description")):
                        station_data.land_cover = str(row["description"])
                    continue
                elif quantity == "climate classification":
                    if pd.notna(row.get("description")):
                        station_data.climate_classification = str(
                            row["description"])
                    continue

                # Skip rows without depth info (for soil properties)
                if pd.isna(depth_from) or pd.isna(depth_to):
                    continue

                depth_key = f"{depth_from:.2f}_{depth_to:.2f}"

                if depth_key not in soil_props:
                    soil_props[depth_key] = ISMNSoilProperties(
                        depth_from_m=float(depth_from),
                        depth_to_m=float(depth_to)
                    )

                # Map quantity names to properties
                if quantity == "saturation" and pd.notna(value):
                    soil_props[depth_key].saturation = float(value)
                elif quantity == "clay fraction" and pd.notna(value):
                    soil_props[depth_key].clay_fraction = float(value)
                elif quantity == "sand fraction" and pd.notna(value):
                    soil_props[depth_key].sand_fraction = float(value)
                elif quantity == "silt fraction" and pd.notna(value):
                    soil_props[depth_key].silt_fraction = float(value)
                elif quantity == "organic carbon" and pd.notna(value):
                    soil_props[depth_key].organic_carbon = float(value)

            station_data.soil_properties = soil_props

        except Exception as e:
            logger.warning(
                f"Failed to load static variables from {file_path}: {e}")

    def load_network(
        self,
        network: str,
        dataset_path: Optional[Path] = None,
        filter_quality: bool = True,
        aggregate_daily: bool = True
    ) -> Dict[str, ISMNStationData]:
        """
        Load all stations in a network.

        Args:
            network: Network name (e.g., "AMMA-CATCH", "TAHMO")
            dataset_path: Path to dataset folder
            filter_quality: Filter to G/M quality flags
            aggregate_daily: Aggregate to daily values

        Returns:
            Dict mapping station names to ISMNStationData objects
        """
        if dataset_path is None:
            datasets = self.list_available_datasets()
            if not datasets:
                raise FileNotFoundError("No ISMN datasets found")
            dataset_path = datasets[0]

        network_path = dataset_path / network
        if not network_path.exists():
            raise FileNotFoundError(f"Network not found: {network}")

        stations = {}
        for station_folder in network_path.iterdir():
            if not station_folder.is_dir():
                continue

            try:
                station_data = self.load_station(
                    station_folder,
                    filter_quality=filter_quality,
                    aggregate_daily=aggregate_daily
                )
                stations[station_data.station] = station_data
                logger.info(f"Loaded station: {station_data.station_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to load station {station_folder.name}: {e}")

        return stations

    def load_all_stations(
        self,
        dataset_path: Optional[Path] = None,
        filter_quality: bool = True,
        aggregate_daily: bool = True
    ) -> Dict[str, ISMNStationData]:
        """
        Load all stations from all networks in a dataset.

        Args:
            dataset_path: Path to dataset folder
            filter_quality: Filter to G/M quality flags
            aggregate_daily: Aggregate to daily values

        Returns:
            Dict mapping station_id to ISMNStationData objects
        """
        if dataset_path is None:
            datasets = self.list_available_datasets()
            if not datasets:
                raise FileNotFoundError("No ISMN datasets found")
            dataset_path = datasets[0]

        all_stations = {}
        networks = self.list_networks(dataset_path)

        for network in networks:
            try:
                network_stations = self.load_network(
                    network,
                    dataset_path=dataset_path,
                    filter_quality=filter_quality,
                    aggregate_daily=aggregate_daily
                )
                for station_data in network_stations.values():
                    all_stations[station_data.station_id] = station_data
            except Exception as e:
                logger.warning(f"Failed to load network {network}: {e}")

        return all_stations


def load_ismn_station(
    station_path: Path,
    filter_quality: bool = True,
    aggregate_daily: bool = True
) -> ISMNStationData:
    """
    Convenience function to load a single ISMN station.

    Args:
        station_path: Path to station folder
        filter_quality: Filter to G/M quality flags
        aggregate_daily: Aggregate to daily values

    Returns:
        ISMNStationData object
    """
    loader = ISMNStationLoader()
    return loader.load_station(
        station_path,
        filter_quality=filter_quality,
        aggregate_daily=aggregate_daily
    )


def get_daily_soil_moisture(
    station_data: ISMNStationData,
    depth_cm: Optional[float] = None
) -> pd.DataFrame:
    """
    Extract daily soil moisture timeseries from station data.

    Args:
        station_data: Loaded ISMNStationData object
        depth_cm: Specific depth to filter to (optional)

    Returns:
        DataFrame with date and soil moisture columns
    """
    if station_data.daily_data is None or station_data.daily_data.empty:
        return pd.DataFrame()

    df = station_data.daily_data.copy()

    if depth_cm is not None:
        # Find closest depth
        available_depths = df["depth_cm"].unique()
        closest_depth = min(available_depths, key=lambda x: abs(x - depth_cm))
        df = df[df["depth_cm"] == closest_depth]

    return df


if __name__ == "__main__":
    # Demo usage
    import sys

    logging.basicConfig(level=logging.INFO)

    # Try to find and load some data
    data_dir = Path("/home/viv/SMPS/data/ismn")
    if not data_dir.exists():
        print("ISMN data directory not found")
        sys.exit(1)

    loader = ISMNStationLoader(data_dir)

    datasets = loader.list_available_datasets()
    print(f"Found {len(datasets)} ISMN datasets")

    if datasets:
        dataset = datasets[0]
        networks = loader.list_networks(dataset)
        print(f"Networks: {networks}")

        if networks:
            network = networks[0]
            stations = loader.list_stations(network, dataset)
            print(f"Stations in {network}: {stations[:5]}...")

            if stations:
                station_path = dataset / network / stations[0]
                print(f"\nLoading station: {station_path}")

                station_data = loader.load_station(station_path)
                print(f"  Station ID: {station_data.station_id}")
                print(
                    f"  Location: ({station_data.latitude:.4f}, {station_data.longitude:.4f})")
                print(f"  Elevation: {station_data.elevation}m")
                print(
                    f"  Available depths: {station_data.available_depths_cm} cm")
                print(f"  Sensors: {list(station_data.sensors.keys())}")
                print(
                    f"  Soil properties: {list(station_data.soil_properties.keys())}")

                if station_data.daily_data is not None:
                    print(
                        f"\n  Daily data shape: {station_data.daily_data.shape}")
                    print(
                        f"  Date range: {station_data.daily_data['date'].min()} to {station_data.daily_data['date'].max()}")
                    print(f"\n  Sample data:")
                    print(station_data.daily_data.head(10))

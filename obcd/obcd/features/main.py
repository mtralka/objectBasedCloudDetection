import logging
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


try:
    import gdal
    import gdalconst
except ImportError:
    from osgeo import gdal
    from osgeo import gdalconst

import numpy as np
from obcd.classes import PlatformData
from obcd.features.extractors.demdata import DEMData
from obcd.features.extractors.demdata import create_mapzen_dataset
from obcd.features.extractors.demdata import extract_dem_data
from obcd.platforms import Landsat8
from obcd.platforms import Sentinel2
from obcd.utils import create_outfile_dataset
from obcd.utils import open_raster_as_array
from obcd.utils import validate_path
from obcd.utils import write_array_to_ds
import pandas as pd
from scipy.ndimage import labeled_comprehension
from scipy.ndimage import mean
from skimage.segmentation import quickshift


logging.basicConfig(filename="logging.log", level="CRITICAL")
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Feature Extractor

    Attributes
    ----------
    `infile` : Union[Path, str]
        Path to Sentinel-2 or Landast-8 file EX. {*._MTL.txt, MTD_*.xml}

    Properties
    -------
    `temp_dirname` : str
        Full path to temporary directory - `self.platform_data.scene_id` + '_temp'

    Methods
    -------
    `run()`
        Run fmask routine. Generates np.ndarray array `self.results` with value code as described with attributes `water_value`, `snow_value`, `cloud_shadow_value`, `cloud_value`
    `save_to_sqlite(db_name: str = "OBCD-features.db", table_name: Optional[str] = None)`
        Save `feature_df` to
    `save_to_csv()`
        ss


    Example
    -------
    >>>
    """

    def __init__(
        self,
        infile: Union[Path, str],
        biome_path: Optional[Union[Path, str]] = None,
        labels_path: Optional[Union[Path, str]] = None,
        temp_dir: Optional[Union[Path, str]] = None,
        save_labels: bool = True,
        kernel_size: Union[int, float] = 1,
        max_distance: Union[int, float] = 7,
        ratio: float = 0.75,
        convert2lab: bool = False,
        auto_run: bool = True,
    ):

        file_prefixes: Tuple[str] = ("_MTL", "MTD_")
        infile: Path = validate_path(infile, check_exists=True, check_is_file=False)
        if infile.is_dir():
            for file in infile.iterdir():
                if any((x in file.name for x in file_prefixes)) and file.suffix in [".txt", ".xml"]:
                    self.infile = infile / file
                    break
            else:
                raise ValueError(
                    "`infile` not detected in dir, please manually pass direct filepath"
                )
        elif infile.is_file():
            if infile.suffix != ".txt" and infile.suffix != ".xml":
                raise ValueError("`infile` must be `.txt` or `.xml`")
            self.infile = infile
        else:
            raise ValueError("`infile` must be file or directory")

        matching_biome_paths: List[Path] = list(
            self.infile.parent.glob("*_fixedmask.img")
        )
        self.biome_path: Optional[Path]
        if biome_path:
            self.biome_path = validate_path(
                biome_path, check_exists=True, check_is_file=True
            )
        elif len(matching_biome_paths) > 0:
            self.biome_path = matching_biome_paths[0]
        else:
            self.biome_path = None

        self.temp_dir: Optional[Path] = (
            validate_path(temp_dir, check_exists=True, check_is_dir=True)
            if temp_dir is not None
            else temp_dir
        )

        self.labels_path: Optional[Path] = (
            validate_path(labels_path, check_exists=True, check_is_file=True)
            if labels_path
            else None
        )

        self.save_labels: bool = save_labels
        self.auto_run: bool = auto_run

        self.supported_platforms: Dict[str, Any] = {"Landsat8": Landsat8, "Sentinel2" : Sentinel2}

        self.quickshift_kwargs: Dict[str, Union[float, int, bool]] = {
            "kernel_size": kernel_size,
            "max_dist": max_distance,
            "ratio": ratio,
            "convert2lab": convert2lab,
        }

        self.platform_data: PlatformData
        self.labels: np.ndarray
        self.feature_df: pd.DataFrame

        if self.auto_run:
            self.run()

    @property
    def root_dir(self) -> Path:
        return self.infile.parent if self.infile.is_dir() else self.infile.parent.parent

    def run(self) -> None:

        self.platform_data = self._extract_platform_data()

        self.labels = self._run_quickshift()

        self.feature_df = pd.DataFrame(
            {
                "segment": np.unique(self.labels),
            }
        )

        self._add_metadata_to_df()

        self._add_raster_mean_to_df()

        self._add_dem_mean_to_df()

        if self.biome_path is not None:
            self._add_biome_mean_to_df()

    def _extract_platform_data(self) -> PlatformData:
        """Extract platform data"""

        for name, platform_object in self.supported_platforms.items():
            if platform_object.is_platform(self.infile):

                logging.info("Identified as %s", name)

                return platform_object.get_data(self.infile)

        logger.error("Platform not found or supported", stack_info=True)
        raise ValueError("Platform not found or supported")

    def _run_quickshift(self) -> np.ndarray:
        """Run quickshift algorithm"""

        if self.labels_path:
            logger.info("Using given labels")
            return np.load(self.labels_path)

        labels = quickshift(
            np.stack(list(self.platform_data.band_data.values()), axis=2),
            **self.quickshift_kwargs,
        )

        if self.save_labels:
            labels_path: Path = (
                self.root_dir / f"{str(self.platform_data.scene_id)}-labels.tif"
            )
            labels_ds = create_outfile_dataset(
                str(labels_path),
                self.platform_data.x_size,
                self.platform_data.y_size,
                self.platform_data.projection_reference,
                self.platform_data.geo_transform,
                number_bands=1,
                data_type=gdalconst.GDT_UInt32,
            )

            _labels = np.where(self.platform_data.band_data["RED"] == 0, np.nan, labels)
            labels_ds = write_array_to_ds(labels_ds, _labels)

            labels_ds = None

        logger.info("%s Quickshift labels found", len(np.unique(labels)))

        return labels

    def _add_metadata_to_df(self) -> None:

        df_cols = [
            "WRS_ROW",
            "WRS_PATH",
            "DATE_ACQUIRED",
            "SUN_AZIMUTH",
            "SUN_ELEVATION",
            "SCENE_ID",
        ]

        for item in df_cols:
            self.feature_df[item] = self.platform_data.__dict__[item.lower()]

        return None

    def _add_raster_mean_to_df(self) -> None:

        for name, array in self.platform_data.band_data.items():
            self.feature_df[f"{name}-mean"] = mean(
                array, self.labels, index=np.unique(self.labels)
            )

        return None

    def _add_dem_mean_to_df(self) -> None:

        with tempfile.TemporaryDirectory() as tempdir:

            ds = create_mapzen_dataset(
                self.platform_data.projection_reference,
                self.platform_data.x_size,
                self.platform_data.y_size,
                self.platform_data.geo_transform,
                self.platform_data.out_resolution,
                self.platform_data.scene_id,
                self.platform_data.nodata,
                Path(tempdir),
            )

            if ds is None:
                print(f"Mapzen failed for {self.platform_data.scene_id}")
                return None

            data: DEMData = extract_dem_data(
                ds, scene_id=self.platform_data.scene_id, temp_dir=Path(tempdir)
            )
            ds = None

        self.feature_df["DEM-mean"] = mean(data.dem, self.labels)
        self.feature_df["SLOPE-mean"] = mean(data.slope, self.labels)
        self.feature_df["ASPECT-mean"] = mean(data.aspect, self.labels)

    def _add_biome_mean_to_df(self) -> None:
        def _biome_labeler(value: np.ndarray) -> int:

            unique_values, count_values = np.unique(value, return_counts=True)

            total_pixel_area: int = np.sum(count_values)

            row_result: Dict[int, float] = {}
            for label_name, number_pixels in zip(unique_values, count_values):
                area_percent: float = number_pixels / total_pixel_area
                row_result[label_name] = area_percent

            results.append(row_result)

            return 0

        def _find_max_values(row: dict) -> int:
            return int(round(max(row.values()), 2) * 100)

        def _find_max_keys(row: dict) -> str:
            return str(max(row, key=row.get))  # type: ignore

        biome_array: np.ndarray = open_raster_as_array(str(self.biome_path))
        results: List[Dict[int, float]] = []

        _ = labeled_comprehension(
            biome_array,
            self.labels,
            np.unique(self.labels),
            _biome_labeler,
            int,
            -1,
            False,
        )

        # find_max_values = lambda row: int(round(max(row.values()), 2) * 100)
        # find_max_keys = lambda row: str(max(row, key=row.get))

        self.feature_df["BIOME-all"] = results
        self.feature_df["BIOME-percent"] = (
            self.feature_df["BIOME-all"].apply(_find_max_values).astype(float)
        )
        self.feature_df["BIOME-name"] = (
            self.feature_df["BIOME-all"].apply(_find_max_keys).astype(int)
        )

        return None

    def save_to_sqlite(
        self, db_name: str = "OBCD-features.db", table_name: Optional[str] = None
    ) -> None:

        table: str = f"{self.platform_data.scene_id}" if not table_name else table_name

        conn = sqlite3.connect(db_name)

        # sqlite does not support the json dtype of 'BIOME-all'
        out_df: pd.DataFrame = self.feature_df.drop("BIOME-all", axis=1, inplace=False)

        out_df.to_sql(table, conn, if_exists="replace")
        conn.close()

        return None

    def save_to_csv(self, name: Optional[str] = None) -> None:
        # TODO Path and stem check
        outfile_name: Union[Path, str] = (
            self.root_dir / f"{self.platform_data.scene_id}-features.csv"
            if not name
            else name
        )

        self.feature_df.to_csv(outfile_name)

        return None

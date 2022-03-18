from enum import Enum
import logging.config
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from obcd.classes import PlatformData
from obcd.extract.extractors.metadata import extract_metadata
from obcd.platforms.platform_base import PlatformBase


try:
    import gdal
except ImportError:
    from osgeo import gdal


logger = logging.getLogger(__name__)


class Landsat8(PlatformBase):
    class Bands(Enum):
        BLUE = 2
        GREEN = 3
        RED = 4
        NIR = 5
        SWIR1 = 6
        # SWIR2 = 7
        CIRRUS = 9
        BT = 10

    RGB: tuple = (Bands.RED, Bands.GREEN, Bands.BLUE)

    OUT_RESOLUTION: int = 30
    NO_DATA: int = np.nan

    @staticmethod
    def is_platform(file_path: Union[Path, str]) -> bool:

        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        file_name = file_path.name

        if ("LC" in file_name) & (
            ("_MTL.txt" in file_name) or ("_MTL.xml" in file_name)
        ):
            return True
        return False

    @classmethod
    def _get_calibration_parameters(cls, file_path: Path) -> dict:

        attributes: list = ["REFLECTANCE_MULT_BAND_{}", "REFLECTANCE_ADD_BAND_{}"]
        target_attributes: list = [
            "SUN_ELEVATION",
            "K1_CONSTANT_BAND_10",
            "K2_CONSTANT_BAND_10",
            "RADIANCE_ADD_BAND_10",
            "RADIANCE_MULT_BAND_10",
            "SUN_AZIMUTH",
            "WRS_PATH",
            "WRS_ROW",
        ]

        for target in attributes:
            for band in cls.Bands.__members__.values():
                if band == cls.Bands.BT:
                    continue
                target_attributes.append(target.format(band.value))

        metadata: Dict[str, str] = extract_metadata(file_path, target_attributes)

        logger.debug("Retrieved %s platform metadata parameter(s)", len(metadata))

        return {k: float(v) for k, v in metadata.items()}

    @classmethod
    def _get_file_names(cls, file_path: Path) -> dict:
        # PRODUCT_CONTENTS/
        attributes: list = ["FILE_NAME_BAND_{}"]
        target_attributes: list = []

        for target in attributes:
            for band in cls.Bands.__members__.values():
                target_attributes.append(target.format(band.value))

        target_attributes.append("LANDSAT_SCENE_ID")
        file_names: dict = extract_metadata(file_path, target_attributes)

        band_names: List[str] = [b.name for b in cls.Bands]

        return {k: v.strip('"') for k, v in zip(band_names, file_names.values())}

    @classmethod
    def get_data(cls, file_path: Union[Path, str]) -> PlatformData:

        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        parameters: Dict[str, Any] = {
            "out_resolution": cls.OUT_RESOLUTION,
            "nodata": cls.NO_DATA,
        }

        calibration = cls._get_calibration_parameters(file_path)

        file_band_names = cls._get_file_names(file_path)

        parameters["scene_id"] = file_band_names.pop(
            "LANDSAT_PRODUCT_ID", file_path.parent.name
        )
        parameters["file_band_names"] = file_band_names
        parameters["sensor"] = "L08_OLI"  # SupportedSensors.L08_OLI
        parameters["sun_azimuth"] = float(calibration.pop("SUN_AZIMUTH"))
        parameters["sun_elevation"] = float(calibration.pop("SUN_ELEVATION"))
        parameters["wrs_path"] = calibration.pop("WRS_PATH")
        parameters["wrs_row"] = calibration.pop("WRS_ROW")
        parameters["date_acquired"] = extract_metadata(file_path, ["DATE_ACQUIRED"])[
            "DATE_ACQUIRED"
        ]
        parameters["calibration"] = calibration

        parameters["band_data"] = {}

        for idx, band in enumerate(cls.Bands.__members__.values()):

            band_number = band.value
            band_name = band.name

            band_path: Path = file_path.parent / file_band_names[band_name]

            band_ds = gdal.Open(str(band_path))

            band_array = band_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)

            logger.debug(
                "Processing band %s - %s from %s", band_number, band_name, band_path
            )

            ##
            # Use first band as projection base
            ##
            if idx == 0:
                parameters["geo_transform"] = band_ds.GetGeoTransform()
                parameters["projection_reference"] = band_ds.GetProjectionRef()

            ##
            # Convert to TOA reflectance
            ##
            if band != cls.Bands.BT:
                processed_band_array = (
                    band_array * calibration[f"REFLECTANCE_MULT_BAND_{band_number}"]
                    + calibration[f"REFLECTANCE_ADD_BAND_{band_number}"]
                )

                processed_band_array = (
                    10000
                    * processed_band_array
                    / np.sin(parameters["sun_elevation"] * np.pi / 180.0)
                )

                # processed_band_array = processed_band_array / 10000.

                # processed_band_array = np.where(
                #     processed_band_array > 1, 1, processed_band_array
                # )
                # processed_band_array = np.where(
                #     processed_band_array < 0, 0, processed_band_array
                # )

            elif band == cls.Bands.BT:

                # convert to TOA
                toa_array: np.ndarray = (
                    band_array * calibration[f"RADIANCE_MULT_BAND_{band_number}"]
                    + calibration[f"RADIANCE_ADD_BAND_{band_number}"]
                )

                # convert to kelvin
                kelvin_array: np.ndarray = (
                    calibration[f"K2_CONSTANT_BAND_{band_number}"]
                ) / np.log(
                    calibration[f"K1_CONSTANT_BAND_{band_number}"] / toa_array + 1
                )

                # convert to celsisus and scale
                celsius_array: np.ndarray = 100 * (kelvin_array - 273.15)

                processed_band_array = celsius_array

            processed_band_array = np.where(
                band_array == 0, cls.NO_DATA, processed_band_array
            ).astype(np.int16)

            parameters["band_data"][band_name] = processed_band_array

            band_ds = None

        parameters["x_size"] = parameters["band_data"]["RED"].shape[1]
        parameters["y_size"] = parameters["band_data"]["RED"].shape[0]

        return PlatformData(**parameters)

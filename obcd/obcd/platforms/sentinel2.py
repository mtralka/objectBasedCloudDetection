from enum import Enum
import logging.config
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
from obcd.classes import PlatformData
from obcd.features.extractors.metadata import extract_metadata
from obcd.platforms.platform_base import PlatformBase
from skimage.measure import block_reduce


try:
    import gdal
except ImportError:
    from osgeo import gdal


logger = logging.getLogger(__name__)


class Sentinel2(PlatformBase):
    class Bands(Enum):
        BLUE = 2
        GREEN = 3
        RED = 4
        NIR = 8
        SWIR1 = 11
        SWIR2 = 12
        CIRRUS = 10

    RGB: tuple = (Bands.RED, Bands.GREEN, Bands.BLUE)
    RESAMPLE_BANDS: tuple = (
        Bands.RED,
        Bands.GREEN,
        Bands.BLUE,
        Bands.NIR,
    )

    OUT_RESOLUTION: int = 30
    NO_DATA: int = -9999

    @staticmethod
    def is_platform(file_path: Union[Path, str]) -> bool:

        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        file_name = file_path.name

        if "MTD_" in file_name:
            return True
        return False

    @classmethod
    def _get_calibration_parameters(cls, file_path: Path) -> dict:

        target_attributes: list = ["AZIMUTH_ANGLE", "ZENITH_ANGLE", "SENSING_TIME"]

        metadata: dict = extract_metadata(file_path, target_attributes)

        logger.debug("Retrieved %s platform metadata parameter(s)", len(metadata))

        return metadata

    @classmethod
    def _get_file_names(cls, file_path: Path) -> Dict[str, Path]:

        attributes: list = ["B{}"]
        target_attributes: dict = {}

        for target in attributes:
            for band in cls.Bands.__members__.values():
                target_attributes[(target.format(str(band.value).zfill(2)))] = band.name

        files_in_target_dir: list = list(
            Path(file_path.parent / "IMG_DATA").glob("*.jp2")
        )

        if len(files_in_target_dir) == 0:
            logger.error("No S2 band files found", stack_info=True)
            raise FileNotFoundError("No S2 band files found")

        file_names: Dict[str, Path] = {}
        for file in files_in_target_dir:
            for band_query, name in target_attributes.items():
                if band_query in file.name:
                    file_names[name] = file
                    break

        if len(file_names) != len(cls.Bands.__members__):
            logger.error("Not enough S2 band files found", stack_info=True)
            raise ValueError("Not enough S2 band files found")

        return file_names

    @classmethod
    def get_data(cls, file_path: Union[Path, str]) -> PlatformData:

        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        parameters: Dict[str, Any] = {
            "out_resolution": cls.OUT_RESOLUTION,
            "nodata": cls.NO_DATA,
        }

        calibration = cls._get_calibration_parameters(file_path)
        file_band_names = cls._get_file_names(file_path)

        parameters["file_band_names"] = file_band_names
        parameters["sensor"] = "S2_MSI"
        parameters["sun_azimuth"] = float(calibration.pop("AZIMUTH_ANGLE"))
        parameters["sun_elevation"] = 90.0 - float(calibration.pop("ZENITH_ANGLE"))
        parameters["date_acquired"] = calibration.pop("SENSING_TIME")
        parameters["calibration"] = calibration

        parameters["scene_id"] = file_path.parent.name  # uses name of parent folder

        parameters["band_data"] = {}

        for band in cls.Bands.__members__.values():

            band_number = band.value
            band_name = band.name

            band_path: Path = file_path.parent / file_band_names[band_name]

            logger.debug(
                "Processing band %s - %s from %s", band_number, band_name, band_path
            )

            ##
            # Downsample CIRRUS to 20m
            ##
            if band == cls.Bands.CIRRUS:
                resample_algorithm: str = "near"
                band_ds = gdal.Warp(
                    "",
                    str(band_path),
                    xRes=30, # TODO
                    yRes=30, # TODO
                    resampleAlg=resample_algorithm,
                    srcNodata=0,
                    dstNodata=0,
                    format="VRT",
                )
            elif band in [cls.Bands.SWIR1, cls.Bands.SWIR2]:
                resample_algorithm: str = "near"
                band_ds = gdal.Warp(
                    "",
                    str(band_path),
                    xRes=10, # TODO
                    yRes=10, # TODO
                    resampleAlg=resample_algorithm,
                    srcNodata=0,
                    dstNodata=0,
                    format="VRT",
                )
            else:
                band_ds = gdal.Open(str(band_path))

            band_array: np.ndarray = np.array(
                band_ds.GetRasterBand(1).ReadAsArray()
            ).astype(np.uint16)

            ##
            # Upsample RGB and NIR to 20m
            ##
            if band in cls.RESAMPLE_BANDS: # TODO from 2
                band_array = block_reduce(band_array, block_size=(3, 3), func=np.mean)

            if band in [cls.Bands.SWIR1, cls.Bands.SWIR2]: # TODO from 2
                band_array = block_reduce(band_array, block_size=(3, 3), func=np.mean)

            ##
            # Use SWIR1 band as projection base
            ##
            if band == cls.Bands.CIRRUS:
                parameters["geo_transform"] = band_ds.GetGeoTransform()
                parameters["projection_reference"] = band_ds.GetProjectionRef()


            ##
            # Saturation of visible bands (RGB)
            ##
            # if parameters.get("vis_saturation") is None:
            #     parameters["vis_saturation"] = np.zeros(band_array.shape).astype(bool)

            # if band in cls.RGB:
            #     parameters["vis_saturation"] = np.where(
            #         band_array == 65535, True, parameters["vis_saturation"]
            #     )

            ##
            # Assign NoData
            ##
            processed_band_array: np.ndarray = np.where(
                band_array > 10000, 10000, band_array
            )
            processed_band_array = np.where(
                band_array == 0, cls.NO_DATA, band_array
            ).astype(np.int16)

            if band == cls.Bands.NIR:
                if processed_band_array.shape == (427, 477):
                    processed_band_array = processed_band_array[:-1, :-1]

            parameters["band_data"][band_name] = processed_band_array

            band_ds = None

        parameters["x_size"] = parameters["band_data"]["RED"].shape[1]
        parameters["y_size"] = parameters["band_data"]["RED"].shape[0]

        return PlatformData(**parameters, wrs_path=-1, wrs_row=-1, )

from os import path
from pathlib import Path
import pickle
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd


try:
    import gdal
    from gdal import Dataset
    import gdalconst
except ImportError:
    from osgeo import gdal
    from osgeo import gdalconst
    from osgeo.gdal import Dataset


def validate_path(
    path: Union[str, Path],
    check_exists: bool = False,
    check_is_file: bool = False,
    check_is_dir=False,
) -> Path:

    valid_path: Path = Path(path) if isinstance(path, str) else path

    if check_exists:
        if not valid_path.exists():
            raise FileExistsError(f"{path} must exist")

    if check_is_file:
        if not valid_path.is_file():
            raise FileNotFoundError(f"{path} must be a file")

    if check_is_dir:
        if not valid_path.is_dir():
            raise ValueError(f"{path} must be a directory")

    return valid_path


def open_raster_as_array(file_path: str, band: int = 1) -> np.ndarray:
    """Opens and returns raster as NumPy array

    Uses GDAL to open raster `band` of raster at `file_path` as NumPy array.
    Verifies the existance of `file_path` and the validity of `band`

    Parameters
    ----------
    file_path : str
        Full file path to target raster file
    band : int
        Band to extract from raster file at `file_path`

    Returns
    -------
    np.ndarray
        n dimensional array of raster `band` from `file_path`
    """

    # Verify that `file_path` is a file before continuing
    if not path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    file_path = str(file_path) if isinstance(file_path, Path) else file_path

    # Open `file_path` with GDAL
    ds = gdal.Open(file_path)

    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to open {file_path}")

    # Count number of bands
    number_bands: int = ds.RasterCount

    # Verify `band` is within the number of bands in `file_path` and
    # greater than zero
    if band > number_bands or band <= 0:
        raise ValueError(f"target band {band} is outside `file_path` band scope")

    # Read `band` as array
    array: np.ndarray = ds.GetRasterBand(band).ReadAsArray()

    # Confirm `band` `array` has data
    if array is None:
        raise TypeError(f"No data present in band {band}. Check raster data")

    # Close file
    ds = None

    return array


def pickle_object(
    name: Union[Path, str], pickle_obj: Optional[Any] = None
) -> Optional[Any]:
    name_path: Path = Path(name) if isinstance(name, str) else name

    if pickle_obj:
        with open(name_path, "wb") as file:
            pickle.dump(pickle_obj, file)
        return None

    if Path(name).is_file():
        with open(name_path, "rb") as file:
            return pickle.load(file)

    raise FileNotFoundError("`name` does not exist or is not a file")


def create_outfile_dataset(
    file_path: str,
    x_size: int,
    y_size: int,
    wkt_projection: str,
    geo_transform: tuple,
    number_bands: int,
    driver: str = "GTiff",
    data_type=gdalconst.GDT_Int16,
    outfile_options: list = ["COMPRESS=DEFLATE"],
) -> Dataset:

    """Creates outfile dataset

    Uses GDAL to create an outfile Dataset to `file_path` using given metadata
    parameters

    Parameters
    ----------
    file_path : str
        Full file path to target raster file
    x_size : int
        Desired outfile dataset X size
    y_size : int
        Desired outfile dataset Y size
    wkt_projection : str
        WKT formated projection for outfile dataset
    geo_transform : tuple
        Geographic transformation for outfile dataset
    number_bands : int
        Number of bands for outfile dataset
    driver : str
        Outfile driver type. Default `GTiff`
    data_type : gdalconst.*
        Outfile data type. Default gdalconst.GDT_Int16
    outfile_options : list
        List of GDAL outfile options. Default ['COMPRESS=DEFLATE']

    Returns
    -------
    osgeo.gdal.Dataset
        GDAl dataset with given metdata parameters
    """
    # Create outfile driver
    driver = gdal.GetDriverByName(driver)

    # Create outfile dataset
    ds = driver.Create(
        file_path, x_size, y_size, number_bands, data_type, outfile_options
    )

    # Confirm successful `ds` creation
    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to create {file_path}")

    # Set outfile projection in WKT format
    ds.SetProjection(wkt_projection)

    # Set outfile geo transform
    ds.SetGeoTransform(geo_transform)

    return ds


def write_array_to_ds(
    ds: Dataset, array: np.ndarray, band: int = 1, no_data_value: int = -9999
) -> Dataset:

    """Writes NumPy array to GDAL Dataset band

    Uses GDAL to write `array` to `ds` `band` using given metadata parameters

    Parameters
    ----------
    ds : osgeo.gdal.Dataset
        GDAL dataset
    array : np.ndarray
        Full file path to target raster file
    band : int
        Target DS band to write `array`
    no_data_value : int
        No data value for `band`. Default -9999

    Returns
    -------
    osgeo.gdal.Dataset
        GDAl dataset with `array` written to `band`

    """
    # Confirm `ds` is valid
    if ds is None:
        raise ValueError(f"`ds` is None")

    number_bands: int = ds.RasterCount

    # Verify `band` is within the number of bands in `file_path` and
    # greater than zero
    if band > number_bands or band <= 0:
        raise ValueError(f"target band {band} is outside `ds` band scope")

    # Write `array` to outfile dataset
    ds.GetRasterBand(band).WriteArray(array)

    # Set outfile `no_data_value`
    ds.GetRasterBand(band).SetNoDataValue(no_data_value)

    return ds


def get_raster_metadata(file_path: str) -> dict:
    """Opens and returns raster metadata

    Uses GDAL to open raster at `file_path` and returns raster metadata. Band
    agnostic

    Parameters
    ----------
    file_path : str
        Full file path to target raster file

    Returns
    -------
    dict
        dictionary of metadata from `file_path` raster
    """
    # Verify that `file_path` is a file before continuing
    if not path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    # Open `file_path` with GDAL
    ds = gdal.Open(file_path)

    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to open {file_path}")

    # Create `stats` dictionary
    stats: dict = {}
    stats["total_bands"] = ds.RasterCount
    stats["x_size"] = ds.RasterXSize
    stats["y_size"] = ds.RasterYSize
    stats["wkt_projection"] = ds.GetProjectionRef()
    stats["geo_transform"] = ds.GetGeoTransform()
    stats["xmin"] = ds.GetGeoTransform()[0]
    stats["xmax"] = stats["xmin"] + stats["x_size"] * ds.GetGeoTransform()[1]
    stats["ymax"] = ds.GetGeoTransform()[3]
    stats["ymin"] = stats["ymax"] + stats["y_size"] * ds.GetGeoTransform()[5]

    # Close file
    ds = None

    return stats

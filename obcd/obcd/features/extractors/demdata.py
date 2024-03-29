from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import subprocess
from typing import NamedTuple
from typing import Union

import numpy as np


try:
    import gdal
    import osr
except ImportError:
    from osgeo import gdal
    from osgeo import osr


##
# Types needed for Aux extraction
##


@dataclass
class DEMData:
    dem: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray


class Coordinate(NamedTuple):
    x: float
    y: float


class BoundingBox(NamedTuple):
    NORTH: float
    EAST: float
    SOUTH: float
    WEST: float


def create_mapzen_dataset(
    projection_reference: tuple,
    x_size: int,
    y_size: int,
    geo_transform: tuple,
    out_resolution: int,
    scene_id: str,
    no_data: Union[int, float],
    temp_dir: Path,
):
    """Create Mapzen GDAL DS using MAPZEN WMS file"""

    with resources.path("obcd.features.extractors", "mapzen_wms.xml") as p:
        mapzen_wms: str = str(p)

    if not Path(mapzen_wms).is_file():
        print("No MAPZEN WMS file found")
        return None

    ##
    # Read DEM WMS
    ##
    dem_ds = gdal.Open(mapzen_wms)

    if not dem_ds:
        print("MAPZEN failed")
        return None

    # original_upper_left: Coordinate = Coordinate(geo_transform[0], geo_transform[3])
    # original_lower_right: Coordinate = Coordinate(
    #     geo_transform[0] + out_resolution * x_size,
    #     geo_transform[3] - out_resolution * y_size,
    # )

    # IF NOT WORK COMMENT
    WGS_84 = osr.SpatialReference()
    WGS_84.ImportFromEPSG(4326)

    # transform to lat / lon
    proj_ref_lat_lon = osr.SpatialReference()
    proj_ref_lat_lon.ImportFromWkt(projection_reference)

    outfile_path: Path = temp_dir / f"_{scene_id}DEM.tif"

    # print(original_upper_left.x,
    #         original_lower_right.y,
    #         original_lower_right.x,
    #         original_upper_left.y,)

    xmin = geo_transform[0]
    ymax = geo_transform[3]
    ymin = ymax + y_size * geo_transform[5]
    xmax = xmin + x_size * geo_transform[1]

    # proj_ref = osr.SpatialReference()
    # proj_ref.ImportFromWkt(projection_reference)

    # coord_original = dict()
    # coord_original['ul'] = (geo_transform[0], geo_transform[3])
    # coord_original['lr'] = (geo_transform[0] + out_resolution * x_size,
    #                         geo_transform[3] - out_resolution * y_size)

    # ds = gdal.Warp(str(outfile_path), dem_ds, dstSRS=proj_ref,
    #                          xRes=out_resolution, yRes=out_resolution, resampleAlg="bilinear",
    #                          outputBounds=(coord_original['ul'][0], coord_original['lr'][1],
    #                                        coord_original['lr'][0], coord_original['ul'][1]),
    #                          srcNodata=no_data, dstNodata=no_data,
    #                          format='GTiff')

    ds = gdal.Warp(
        str(outfile_path),
        dem_ds,
        # srcSRS="EPSG:3857",
        dstSRS=proj_ref_lat_lon,
        # dstSRS=projection_reference,
        xRes=out_resolution,
        yRes=out_resolution,
        resampleAlg="bilinear",
        outputBounds=(xmin, ymin, xmax, ymax),
        srcNodata=no_data,
        dstNodata=no_data,
        format="GTiff",
    )

    # ds = gdal.Warp(
    #     '',
    #     dem_ds,
    #     dstSRS=proj_ref_lat_lon,
    #     xRes=out_resolution,
    #     yRes=out_resolution,
    #     resampleAlg="bilinear",
    #     outputBounds=(
    #        xmin,
    #        ymin,
    #        xmax,
    #        ymax
    #     ),
    #     srcNodata=no_data,
    #     dstNodata=no_data,
    #     format="VRT",
    # )

    # ds = gdal.Warp(
    #     str(outfile_path),
    #     dem_ds,
    #     dstSRS=proj_ref_lat_lon,
    #     xRes=out_resolution,
    #     yRes=out_resolution,
    #     resampleAlg="bilinear",
    #     outputBounds=(
    #         original_upper_left.x,
    #         original_lower_right.y,
    #         original_lower_right.x,
    #         original_upper_left.y,
    #     ),
    #     srcNodata=no_data,
    #     dstNodata=no_data,
    #     format="GTiff",
    # )

    if not ds:
        print("MAPZEN failed")
        return None

    ds.GetRasterBand(1).SetNoDataValue(no_data)
    dem_ds = None

    return ds


def extract_dem_data(ds, scene_id: str, temp_dir: Path) -> DEMData:

    dem_arr: np.ndarray = ds.GetRasterBand(1).ReadAsArray()

    slope_name: str = f"{scene_id}_slope.tif"
    slope_ds = gdal.DEMProcessing(
        str(temp_dir / slope_name), ds, processing="slope", slopeFormat="degree"
    )
    slope_arr: np.ndarray = slope_ds.GetRasterBand(1).ReadAsArray()
    slope_ds = None

    aspect_name: str = f"{scene_id}_aspect.tif"
    aspect_ds = gdal.DEMProcessing(
        str(temp_dir / aspect_name), ds, processing="aspect", zeroForFlat=True
    )
    aspect_arr: np.ndarray = aspect_ds.GetRasterBand(1).ReadAsArray()
    aspect_ds = None

    ds = None

    return DEMData(dem=dem_arr, slope=slope_arr, aspect=aspect_arr)

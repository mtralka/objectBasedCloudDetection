from copyreg import pickle
from pathlib import Path
import subprocess
import tempfile
from typing import Optional
from typing import Union

import geopandas as gpd
import numpy as np
from obcd.classes import BaseModel
from obcd.utils import get_raster_metadata
from obcd.utils import pickle_object
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn_pandas import DataFrameMapper


try:
    import gdal
    import gdalconst
except ImportError:
    from osgeo import gdal
    from osgeo import gdalconst


class Predict(BaseModel):
    def __init__(
        self,
        model: Union[Path, str, MLPClassifier],
        features: Union[str, Path, pd.DataFrame],
        labels: Union[Path, str],
        scaler: Union[Path, str],
        read_pkl: bool = True,
        auto_run: bool = True,
    ) -> None:

        if isinstance(model, MLPClassifier):
            self.clf = model
        elif isinstance(model, (Path, str)):
            if not read_pkl:
                raise PermissionError(
                    "Not allowed to read `pkl` files. `read_pkl` must be True"
                )
            self.clf = pickle_object(model)
        else:
            raise TypeError(
                "`model` must be `MLPClassifier` or a `Path` / `str` `.pkl` object"
            )

        self.read_pkl: bool = read_pkl
        self.labels: Path = Path(labels) if isinstance(labels, str) else labels
        self.feature_df: pd.DataFrame = self._open_feature_data(features, read_pkl)
        self.scaler: Path = Path(scaler) if isinstance(scaler, str) else scaler

        self.X_arr: np.ndarray
        self.y_predicted: np.ndarray
        self.y: Optional[np.ndarray] = None

        if auto_run:
            self.run()

    def run(self) -> None:
        # self.feature_df = self._prepare_data(self.feature_df)

        self._engineer_features()
        self.predict()

    def _engineer_features(self) -> None:
        # X: pd.DataFrame = self.feature_df.drop(
        #     ["segment", "CLOUD", "SCENE_ID", "WRS_ROW", "WRS_PATH"],
        #     axis=1,
        #     errors="ignore",
        # )

        X: pd.DataFrame = self.feature_df[
            self.SPECTRAL_COLS
        ]

        if "CLOUD" in self.feature_df.columns:
            self.y = self.feature_df["CLOUD"]

        scaler = pickle_object(self.scaler)

        self.X_arr = scaler.transform(X.to_numpy())

    def predict(
        self, data: Optional[Union[str, Path, pd.DataFrame]] = None
    ) -> np.ndarray:

        if data is not None:
            self.feature_df: pd.DataFrame = self._open_feature_data(data, self.read_pkl)
            self.feature_df = self._prepare_data(self.feature_df)
            self._engineer_features()

        self.y_predicted: np.ndarray = self.clf.predict(self.X_arr)

        # self.X["segment"] = self.feature_df["segment"]
        # self.X["PRED_CLOUD"] = self.y_predicted
        self.feature_df["PRED_CLOUD"] = self.y_predicted

        if self.y is not None:
            # self.X["CLOUD"] = self.y
            print(
                "Test OA = %.3f%%"
                % (100.0 * np.sum(self.y_predicted == self.y) / self.y.shape[0])
            )

    def save_prediction_to_csv(self):
        ...

    def save_prediction_to_raster(self):

        with tempfile.TemporaryDirectory() as tempdir:
            rasterized_name: Path = self.labels.parent / "rasterized_output.tif"
            vectorized_name: Path = Path(tempdir) / "vectorized.shp"
            vectorized_joined_name: Path = Path(tempdir) / "vectorized_joined.shp"

            raster_metadata: dict = get_raster_metadata(str(self.labels))

            subprocess.run(
                [
                    "gdal_polygonize",
                    str(self.labels),
                    "-b",
                    "1",
                    "-f",
                    "ESRI Shapefile",
                    str(vectorized_name),
                    "labels",
                    "DN",
                ],
                shell=True,
            )

            gdf: gpd.GeoDataFrame = gpd.read_file(str(vectorized_name))

            gdf = gdf.merge(self.feature_df, left_on="DN", right_on="segment")
            self.gdf = gdf

            gdf[["DN", "geometry", "PRED_CLOUD"]].to_file(str(vectorized_joined_name))

            subprocess.run(
                [
                    "gdal_rasterize",
                    "-a",
                    "PRED_CLOUD",
                    "-ts",
                    str(round(float(raster_metadata["x_size"]))),
                    str(round(float(raster_metadata["y_size"]))),
                    "-a_nodata",
                    "0",
                    "-te",
                    str(raster_metadata["xmin"]),
                    str(raster_metadata["ymin"]),
                    str(raster_metadata["xmax"]),
                    str(raster_metadata["ymax"]),
                    "-ot",
                    "Byte",
                    "-of",
                    "Gtiff",
                    str(vectorized_joined_name),
                    str(rasterized_name),
                ],
                shell=True,
            )

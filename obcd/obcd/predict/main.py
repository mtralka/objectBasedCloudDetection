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
from skimage import morphology
from sklearn.neural_network import MLPClassifier
from sklearn_pandas import DataFrameMapper


try:
    import gdal
    import gdalconst
    import ogr
except ImportError:
    from osgeo import gdal
    from osgeo import gdalconst
    from osgeo import ogr


class Predict(BaseModel):
    def __init__(
        self,
        model: Union[Path, str, MLPClassifier],
        features: Union[str, Path, pd.DataFrame],
        labels: Optional[Union[Path, str]] = None,
        scaler: Optional[Union[Path, str]] = None,
        read_pkl: bool = True,
        auto_run: bool = True,
    ) -> None:

        ##
        # Model
        ##
        self._model_path: Optional[Path] = None
        self.read_pkl: bool = read_pkl
        if isinstance(model, MLPClassifier):
            self.clf = model
        elif isinstance(model, (Path, str)):
            if not read_pkl:
                raise PermissionError(
                    "Not allowed to read `pkl` files. `read_pkl` must be True"
                )
            self.clf = pickle_object(model)
            self._model_path = Path(model) if isinstance(model, str) else model
        else:
            raise TypeError(
                "`model` must be `MLPClassifier` or a `Path` / `str` `.pkl` object"
            )

        ##
        # Scaler
        ##
        self.scaler: Path
        if isinstance(scaler, (str, Path)):
            self.scaler = Path(scaler) if isinstance(scaler, str) else scaler
        elif scaler is None and self._model_path is not None:

            possible_scaler: Path = (
                self._model_path.parent / f"{self._model_path.stem}_scaler.pkl"
            )

            if not possible_scaler.is_file():
                raise ValueError(
                    "`scaler` auto-detect failed, please pass `scaler` path manually"
                )

            self.scaler = possible_scaler
        else:
            raise ValueError("`scaler` must be `str`, `Path`, or `None`")

        ##
        # Features
        ##
        self._feature_path: Optional[Path] = None
        if isinstance(features, (str, Path)):
            self._feature_path = (
                Path(features) if isinstance(features, str) else features
            )
        self.feature_df: pd.DataFrame = self._open_feature_data(features, read_pkl)

        ##
        # Labels
        ##
        self.labels: Path
        if labels:
            self.labels = Path(labels) if isinstance(labels, str) else labels
        elif isinstance(features, (Path, str)):
            possible_label: Path = (
                self._feature_path.parent
                / self._feature_path.name.replace("features.csv", "labels.tif")
            )
            if not possible_label.is_file():
                raise ValueError(
                    "`label` auto-detect failed, please pass `label` path manually"
                )
            self.labels = possible_label
        else:
            raise ValueError(
                "`label` must be `str`, `Path`, or `None` (for auto-detect)"
            )

        self.X_arr: np.ndarray
        self.y_predicted: np.ndarray
        self.y: Optional[np.ndarray] = None

        if auto_run:
            self.run()

    @property
    def scene_id(self):

        return self.feature_df["SCENE_ID"].iloc[0]

        # if self._model_path is not None:
        #     return self._feature_path.name.replace("features.csv", "")
        # return 

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

        X: pd.DataFrame = self.feature_df[self.SPECTRAL_COLS]

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
        self.feature_df["PROB_CLOUD"] = self.clf.predict_proba(self.X_arr)[:,1]

        if self.y is not None:
            # self.X["CLOUD"] = self.y
            print(
                "Test OA = %.3f%%"
                % (100.0 * np.sum(self.y_predicted == self.y) / self.y.shape[0])
            )

    def save_prediction_to_csv(self):
        predicted_name: Path = self.labels.parent / f"{self.scene_id}-predicted.csv"
        
        self.feature_df.to_csv(predicted_name)

    def save_prediction_to_raster(self):

        with tempfile.TemporaryDirectory() as tempdir:
            rasterized_name: Path = self.labels.parent / f"{self.scene_id}-cloud_mask.tif"
            vectorized_name: Path = Path(tempdir) / "vectorized.shp"
            vectorized_joined_name: Path = Path(tempdir) / "vectorized_joined.shp"

            raster_metadata: dict = get_raster_metadata(str(self.labels))

            # subprocess.run(
            #     [
            #         "gdal_polygonize",
            #         str(self.labels),
            #         "-b",
            #         "1",
            #         "-f",
            #         "ESRI Shapefile",
            #         str(vectorized_name),
            #         "labels",
            #         "DN",
            #     ],
            #     shell=True,
            # )
            # print("done polyg")

            labels_ds = gdal.Open(str(self.labels))

            driver = ogr.GetDriverByName("ESRI Shapefile")
            out_ds = driver.CreateDataSource(str(vectorized_name))
            out_layer = out_ds.CreateLayer("labels", srs=labels_ds.GetSpatialRef())

            field = ogr.FieldDefn("DN", ogr.OFTInteger)
            out_layer.CreateField(field)

            gdal.Polygonize(labels_ds.GetRasterBand(1), None, out_layer, 0, [])

            labels_ds = None
            out_ds = None
            out_layer = None
            field = None
            out_layer = None

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

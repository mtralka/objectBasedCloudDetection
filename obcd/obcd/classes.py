from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd

from obcd.utils import pickle_object


@dataclass
class PlatformData:
    out_resolution: int
    x_size: int
    y_size: int
    sensor: str
    scene_id: str
    sun_elevation: Union[float, int]
    sun_azimuth: Union[float, int]
    geo_transform: tuple
    projection_reference: tuple
    calibration: Any
    file_band_names: List[str]
    band_data: Dict[str, np.ndarray]
    wrs_path: int
    wrs_row: int
    date_acquired: str
    nodata: Union[int, float]


class BaseModel:

    HOMOGONEITY_THRESHOLD: float = 0.2

    SPECTRAL_COLS: List[str] = [
        "RED-mean",
        "BLUE-mean",
        "GREEN-mean",
        "NIR-mean",
        "SWIR1-mean",
        "CIRRUS-mean",
        "BT-mean",
        "DEM-mean",
        "SLOPE-mean",
        "ASPECT-mean",
    ]

    TRAINING_COLS: List[str] = ["BIOME-percent", "BIOME-all", "BIOME-name"]

    @classmethod
    def _prepare_data(cls, df: pd.DataFrame) -> pd.DataFrame:

        prepared_df: pd.DataFrame = df

        prepared_df.dropna(inplace=True)

        # convert date to numeric
        prepared_df["DATE_ACQUIRED"] = prepared_df["DATE_ACQUIRED"].str.replace("-", "")

        # drop rows with spectral nodata
        for col in cls.SPECTRAL_COLS:
            prepared_df = prepared_df[prepared_df[col] != 0.0]
            prepared_df = prepared_df[prepared_df[col] != -9999.0]
            prepared_df = prepared_df[prepared_df[col] != np.nan]
            prepared_df = prepared_df[prepared_df[col] != np.inf]

        if "BIOME-percent" in prepared_df.columns:
            # only use objects with 60% or higher homogeneity
            prepared_df = prepared_df[
                prepared_df["BIOME-percent"] >= cls.HOMOGONEITY_THRESHOLD
            ]

            prepared_df.drop("BIOME-percent", axis=1, inplace=True)

        if "BIOME-all" in prepared_df.columns:
            prepared_df.drop("BIOME-all", axis=1, inplace=True)

        if "BIOME-name" in prepared_df.columns:
            prepared_df["CLOUD"] = np.where(
                (prepared_df["BIOME-name"] == 128) | (prepared_df["BIOME-name"] == 64),
                0,
                1,
            )
            # prepared_df["CLOUD"] = np.where(prepared_df["BIOME-name"]== 128, 0, 1)
            prepared_df.drop("BIOME-name", axis=1, inplace=True)

        return prepared_df

    @classmethod
    def _open_feature_data(
        cls, data: Union[str, Path, pd.DataFrame], read_pkl: bool
    ) -> pd.DataFrame:
        dataframe: pd.DataFrame
        if isinstance(data, pd.DataFrame):
            dataframe = data
        elif isinstance(data, str):
            dataframe = cls.__open_data(Path(data), read_pkl)
        elif isinstance(data, Path):
            dataframe = cls.__open_data(data, read_pkl)
        else:
            raise ValueError(
                "`data` must be `pd.DataFrame`, or filepath / directory `str` / `Path`"
            )

        return dataframe

    @classmethod
    def __open_data(cls, data_path: Path, read_pkl: bool) -> pd.DataFrame:
        return_df: pd.DataFrame
        if data_path.is_file():
            file_suffix = data_path.suffix
            if file_suffix == ".db":

                conn = sqlite3.connect(data_path)
                res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables: List[str] = [x[0] for x in res.fetchall()]

                print(f"Detected db with {len(db_tables)} table(s)")

                query: str = "SELECT * from {}"

                temp_db_df: pd.DataFrame = pd.read_sql_query(
                    query.format(db_tables.pop(0)), conn
                ).iloc[:, 1:]

                temp_db_df = cls._prepare_data(temp_db_df)

                for table in db_tables:
                    temp_df: pd.DataFrame = pd.read_sql_query(
                        query.format(table), conn
                    ).iloc[:, 1:]
                    temp_df = cls._prepare_data(temp_df)
                    temp_db_df = pd.concat([temp_db_df, temp_df], ignore_index=True)
                    del temp_df

                return_df = temp_db_df
                del temp_db_df

            elif file_suffix == ".csv":
                return_df = pd.read_csv(data_path).iloc[:, 1:]
                return_df = cls._prepare_data(return_df)
            elif file_suffix == ".pkl":
                if not read_pkl:
                    raise PermissionError(
                        "Not allowed to read `pkl` files. `read_pkl` must be True"
                    )

                return_df = pickle_object(data_path)
                return_df = cls._prepare_data(return_df)
            else:
                raise NotImplementedError("File must be `.csv` or `.db`")
        elif data_path.is_dir():
            folder_file_contents: List[Path] = [
                x for x in data_path.iterdir() if x.is_file() and x.suffix == ".csv"
            ]

            print(f"Detected folder with {len(folder_file_contents)} items")

            temp_folder_df: pd.DataFrame = pd.read_csv(
                folder_file_contents.pop(0)
            ).iloc[:, 1:]

            temp_folder_df = cls._prepare_data(temp_folder_df)

            for file_item in folder_file_contents:
                temp_df: pd.DataFrame = pd.read_csv(file_item).iloc[:, 1:]
                temp_df = cls._prepare_data(temp_df)
                temp_folder_df = pd.concat([temp_folder_df, temp_df], ignore_index=True)
                del temp_df

            return_df = temp_folder_df
            del temp_folder_df
        else:
            raise NotImplementedError("`data` must be a file or folder")

        return return_df

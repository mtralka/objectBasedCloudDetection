from collections import Counter
from enum import Enum
from enum import auto
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import numpy as np
from obcd.classes import BaseModel
from obcd.utils import pickle_object
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


class ResampleTypes(Enum):
    NONE: int = auto()
    SMOTE: int = auto()


class Train(BaseModel):
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        resample: Optional[ResampleTypes] = ResampleTypes.NONE,
        model_path: Union[Path, str] = "pickled_model",
        save_model: bool = False,
        homogeneity_threshold: int = 60,
        test_size: float = 0.3,
        read_pkl: bool = True,
        auto_run: bool = True,
    ) -> None:

        self.training_df: pd.DataFrame = self._open_feature_data(data, read_pkl)

        self.resample: ResampleTypes
        for method_type in ResampleTypes.__members__.values():
            if resample == method_type or resample == method_type.value:
                self.resample = method_type
                break
            elif resample is None:
                self.resample = ResampleTypes.NONE
                break
            elif (
                isinstance(resample, str)
                and resample.upper().strip() == method_type.name
            ):
                self.resample = ResampleTypes[resample.upper().strip()]
                break
            else:
                raise AttributeError(
                    f"{resample} method not recognized. Select from {','.join([item.name for item in ResampleTypes.__members__.values()])}"
                )

        self.homogeneity_threshold = homogeneity_threshold
        self.test_size = test_size
        self.model_path: Optional[Union[Path, str]] = model_path
        self.save_model: bool = save_model

        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_test: np.ndarray
        self.clf: Any

        if auto_run:
            self.run()

    def run(self) -> None:

        # self.training_df = self._prepare_data(self.training_df)
        self._enginer_features()
        self._train()

        if self.save_model:
            self.pickle_model()

    def _enginer_features(self) -> None:
        X: pd.DataFrame = self.training_df.drop(
            ["segment", "CLOUD", "SCENE_ID", "WRS_ROW", "WRS_PATH"], axis=1
        )

        y: pd.DataFrame = self.training_df["CLOUD"]

        print(f"Y Counts:\n{y.value_counts()}")

        X_resampled: np.ndarray
        y_resampled: np.ndarray
        if self.resample == ResampleTypes.SMOTE:
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = SMOTE().fit_resample(X, y)

            print(f"Y resampled Counter: {sorted(Counter(y_resampled).items())}")
            print(f"Y Resampled Counts: {y.value_counts()}")
        elif self.resample == ResampleTypes.NONE:
            X_resampled = X
            y_resampled = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=self.test_size,
            random_state=0,
            stratify=y_resampled,
        )

        scaler = StandardScaler()

        MAPPER = DataFrameMapper(
            [
                (
                    [
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
                    ],
                    scaler,
                )
            ]
        )

        X_train_norm = MAPPER.fit_transform(X_train)
        X_test_norm = MAPPER.transform(X_test)

        self.X_train = X_train_norm
        self.X_test = X_test_norm
        self.y_train = y_train
        self.y_test = y_test

        self.scaler = scaler

    def _train(self) -> None:

        # clf = MLPClassifier(
        #     hidden_layer_sizes=(10,),
        #     max_iter=300,
        #     alpha=1e-2,
        #     solver="adam",
        #     verbose=10,
        #     tol=1e-4,
        #     random_state=1,
        #     learning_rate_init=0.005,
        # )

        clf = MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            solver="adam",
            alpha=0,
            batch_size=128,
            learning_rate_init=0.005,
            shuffle=True,
            validation_fraction=0.5,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=20,
            verbose=10,
        )

        # param_grid = {
        #     'n_estimators': [100, 200, 300, 500]
        #     }
        # random_forest_model = RandomForestClassifier()
        # clf = GridSearchCV(estimator = random_forest_model , param_grid = param_grid, cv = 3, n_jobs = -1)
        # print()

        # clf = RandomForestClassifier(n_estimators=200)

        clf.fit(self.X_train, self.y_train)

        self.clf = clf
        self.score = clf.score(self.X_test, self.y_test)

        # print(self.score)

    def pickle_model(self, path: Optional[Union[Path, str]] = None):
        save_str: Union[str, Path] = path if path else self.model_path
        save_path: Path = Path(save_str) if isinstance(save_str, str) else save_str

        scaler_path: Path = save_path.parent / f"{save_path.stem}_scaler.pkl"

        if save_path.suffix != ".pkl":
            save_path = Path(str(save_path) + ".pkl")

        pickle_object(save_path, self.clf)
        pickle_object(scaler_path, self.scaler)

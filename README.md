# Object Based Cloud Detection - OBCD

🚧 **under construction** 🚧 

*APIs and documentation will change*


Object based cloud detection for Landsat/Sentinel imagery. 


## Latest Results

|       | NC           | CLOUD       |              |
|-------|--------------|-------------|--------------|
| NC    |          515 |         164 | 0.7584683358 |
| CLOUD |           29 |         489 | 0.944015444 |
|       | 0.9466911765 | 0.7488514548 | 0.8387635756 |
 
## Installation

This program uses `poetry` for package management. Note, `GDAL` **is** required but not listed in the `pyproject.toml`. `GDAL` must be installed in the runtime environment.

### Build and install from source

- Ensure poetry is installed

- Navigate to the directory containing `pyproject.toml`

- `poetry install`

- `poetry build`

Follow steps for **Install from pre-built `.whl`** to install the project locally

**or**

The `obcd` script configured within poetry allows for simple CLI access without installation

- `poetry run obcd [ARGS] [KWARGS]`

### Install from pre-built '.whl'

We can achieve the same function result from distributing the project `.whl`s and installing locally through pip/pipx/etc.

- `pip install path/to/.whl`

Now, `obcd` is fully installed and accessible anywhere in the installed env

## Usage

The OBCD program is split into 3 main modules: `FEATURES`, `TRAIN`, and `PREDICT`. 

**Until the API stabilizes, see module docstrings for in-depth documentation**

### Features

The `features` module handles the extraction and creation of object-based features from Landsat L1 scenes (EX `LC0{8/9}_L1TP_PPPRRR_YYYYMMDD_yyyymmdd_02_T1`)


```python
from obcd import FeatureExtractor

folder: Path | str = "path/to/landsat/scene/folder"
extractor = FeatureExtractor(folder)

extractor.save_to_sqlite()
extractor.save_to_csv()

```

### Train

The `train` module handles the training of assorted ML models from object-based features generated with the FeatureExtractor. 

```python
from obcd import Train

target: Path | str = "path/to/{.db/.csv/folder of .csvs}"
trainer = Train(target)

trainer.pickle_model()
```

### Predict

The `predict` module handles the prediction and output generation of trained models from `train`


```python
from obcd import Predict

model_path: Path | str = "path/to/train/model"
features_to_predict: str | Path | pd.DataFrame = "path/to/features"

predictor = Predict(model_path, features_to_predict)
predictor.save_prediction_to_raster()
```

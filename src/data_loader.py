"""CSV schema validation for the baseline, separate from model fitting."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


def read_csv(path: Path, *, id_column: str) -> pd.DataFrame:
    # pandas silently renames duplicate headers; reject them before parsing.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not name.strip() for name in header) or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: missing, blank, or duplicate headers")
    # Preserve leading zeros and literal IDs such as NA instead of guessing numeric IDs.
    frame = pd.read_csv(path, dtype={id_column: str}, keep_default_na=False, na_values=[""])
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError(f"{path.name}: row fields do not match the CSV header")
    if frame.empty:
        raise ValueError(f"{path.name}: dataset is empty")
    return frame


def load_tabular(data_root: Path, config: Config):
    train, test, sample = [
        read_csv(data_root / name, id_column=config.id_column)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    ]
    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        if config.id_column not in frame:
            raise ValueError(f"{name}: missing ID column {config.id_column!r}")
        if frame[config.id_column].isna().any() or frame[config.id_column].duplicated().any():
            raise ValueError(f"{name}: baseline requires non-missing, unique IDs")
    if config.target not in train or train[config.target].isna().any():
        raise ValueError("Train target is missing or contains missing values")
    if config.target in test:
        raise ValueError("Test data contains the target column")
    if set(sample.columns) != {config.id_column, config.output_column}:
        raise ValueError("Baseline supports exactly one ID and one prediction column")
    if set(test[config.id_column]) != set(sample[config.id_column]):
        raise ValueError("Test and sample IDs differ")
    excluded = {config.target, config.id_column, config.split_column}
    features = [column for column in train if column not in excluded]
    test_features = [column for column in test if column not in excluded]
    if not features or set(features) != set(test_features):
        raise ValueError("Train/test feature schemas differ or no usable features remain")
    if config.split_column and config.split_column not in train:
        raise ValueError(f"Missing split column {config.split_column!r}")
    if config.split_column and train[config.split_column].isna().any():
        raise ValueError("Split column contains missing values")
    for name, frame in (("train", train), ("test", test)):
        numeric = frame[features].select_dtypes(include="number")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"{name}: infinite feature values")
    if config.task == "regression":
        if (
            not pd.api.types.is_numeric_dtype(train[config.target])
            or not np.isfinite(train[config.target].to_numpy(dtype=float)).all()
        ):
            raise ValueError("Regression target must contain finite numeric values")
    elif train[config.target].nunique() < 2:
        raise ValueError("Classification needs at least two classes")
    if config.prediction == "probability" and (
        train[config.target].nunique() != 2
        or config.positive_label not in train[config.target].unique()
    ):
        raise ValueError("Probability mode requires two classes and a matching positive_label")
    return train, test, sample, features

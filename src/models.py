"""A lightweight mixed-type baseline with preprocessing fitted inside each split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.select_dtypes(include="number") if frame[column].notna().any()
    ]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.select_dtypes(exclude="number") if frame[column].notna().any()
    ]


def build_model(features: pd.DataFrame, config: Config) -> Pipeline:
    if features.empty:
        raise ValueError("Cannot build a model without feature rows and columns")
    # Callables run during fit, so empty-column selection only sees that partition.
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    estimator = (
        Ridge(solver="lsqr")
        if config.task == "regression"
        else LogisticRegression(max_iter=2000, random_state=config.seed)
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])

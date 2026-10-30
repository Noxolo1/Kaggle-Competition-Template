"""Holdouts with explicit protection against group and timestamp overlap."""

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import Config


def holdout_indices(train: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(train))
    if config.split == "group":
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.validation_size, random_state=config.seed
        )
        fit, valid = next(splitter.split(train, groups=train[config.split_column]))
    elif config.split == "time":
        times = pd.to_datetime(train[config.split_column], errors="raise", utc=True)
        if times.isna().any():
            raise ValueError("Time split contains missing timestamps")
        order = np.argsort(times.to_numpy(), kind="stable")
        boundary = len(train) - math.ceil(len(train) * config.validation_size)
        if boundary <= 0:
            raise ValueError("Not enough rows for a time holdout")
        cutoff = times.iloc[order[boundary]]
        # All tied timestamps stay together; training is strictly earlier.
        fit, valid = indices[(times < cutoff).to_numpy()], indices[(times >= cutoff).to_numpy()]
    else:
        fit, valid = train_test_split(
            indices,
            test_size=config.validation_size,
            random_state=config.seed,
            stratify=train[config.target] if config.task == "classification" else None,
        )
    if len(fit) == 0 or len(valid) == 0:
        raise ValueError("Split produced an empty partition")
    if config.task == "classification":
        classes = set(train[config.target])
        if (
            set(train.iloc[fit][config.target]) != classes
            or set(train.iloc[valid][config.target]) != classes
        ):
            raise ValueError("Each classification partition must contain every class")
    return fit, valid

"""Submission schema, order, and numerical sanity checks before writing CSV."""

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    probability_columns: Sequence[str] = (),
) -> None:
    """Require exact schema and ID order. Duplicate IDs are valid in some competitions."""
    if not sample.columns.is_unique or not submission.columns.is_unique:
        raise ValueError("Submission columns must be unique")
    if sample.empty or list(submission.columns) != list(sample.columns):
        raise ValueError("Submission must match the non-empty sample's column names and order")
    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match sample")
    if len(set(id_columns)) != len(id_columns) or not set(id_columns) <= set(sample.columns):
        raise ValueError("id_columns must be unique sample column names")
    targets = [column for column in sample.columns if column not in id_columns]
    if not targets:
        raise ValueError("Submission must contain prediction columns")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    for column in id_columns:
        if (
            not submission[column]
            .reset_index(drop=True)
            .equals(sample[column].reset_index(drop=True))
        ):
            raise ValueError(f"ID values, dtype, or order differ in {column!r}")
    for column in targets:
        if (
            pd.api.types.is_numeric_dtype(submission[column])
            and not np.isfinite(submission[column].to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"Non-finite predictions in {column!r}")
    if not set(probability_columns) <= set(targets):
        raise ValueError("Probability columns must be prediction columns")
    for column in probability_columns:
        values = submission[column]
        if not pd.api.types.is_numeric_dtype(values) or not values.between(0, 1).all():
            raise ValueError(f"Probabilities outside [0, 1] in {column!r}")


def write_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    path: str | Path,
    *,
    id_columns: Sequence[str],
    probability_columns: Sequence[str] = (),
    overwrite: bool = False,
) -> Path:
    validate_submission(
        submission, sample, id_columns=id_columns, probability_columns=probability_columns
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with destination.open("x", encoding="utf-8", newline="") as handle:
            try:
                submission.to_csv(handle, index=False)
            except BaseException:
                handle.close()
                destination.unlink(missing_ok=True)
                raise
    else:
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                submission.to_csv(handle, index=False)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination

"""Small, strict JSON configuration for the tabular baseline."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    target: str
    id_column: str
    task: str = "regression"
    competition: str = ""
    submission_target: str | None = None
    prediction: str = "label"
    positive_label: str | int | float | None = None
    split: str = "random"
    split_column: str | None = None
    validation_size: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        for key in ("target", "id_column"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a non-empty column name")
        if self.target == self.id_column:
            raise ValueError("target and id_column must differ")
        if not isinstance(self.competition, str):
            raise ValueError("competition must be a string")
        if self.submission_target is not None and (
            not isinstance(self.submission_target, str) or not self.submission_target.strip()
        ):
            raise ValueError("submission_target must be a non-empty string or null")
        if self.output_column == self.id_column:
            raise ValueError("submission_target must differ from id_column")
        if self.task not in ("regression", "classification"):
            raise ValueError("task must be regression or classification")
        if self.prediction not in ("label", "probability"):
            raise ValueError("prediction must be label or probability")
        if self.task == "regression" and self.prediction != "label":
            raise ValueError("regression uses prediction=label (numeric predictions)")
        if self.prediction == "probability" and self.positive_label is None:
            raise ValueError("Set positive_label explicitly for binary probability predictions")
        if self.positive_label is not None and (
            isinstance(self.positive_label, bool)
            or not isinstance(self.positive_label, (str, int, float))
            or isinstance(self.positive_label, float)
            and not math.isfinite(self.positive_label)
        ):
            raise ValueError("positive_label must be a finite number or string")
        if self.prediction != "probability" and self.positive_label is not None:
            raise ValueError("positive_label only applies to probability predictions")
        if self.split not in ("random", "group", "time"):
            raise ValueError("split must be random, group, or time")
        if self.split != "random" and (
            not isinstance(self.split_column, str) or not self.split_column.strip()
        ):
            raise ValueError("group/time splits require split_column")
        if self.split == "random" and self.split_column is not None:
            raise ValueError("split_column requires split=group or time")
        if self.split_column in (self.target, self.id_column):
            raise ValueError("split_column must differ from target and id_column")
        if (
            isinstance(self.validation_size, bool)
            or not isinstance(self.validation_size, (float, int))
            or not 0 < self.validation_size < 1
        ):
            raise ValueError("validation_size must be between 0 and 1")
        # Validation must not change global RNG state.
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")

    @property
    def output_column(self) -> str:
        return self.submission_target or self.target

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate configuration key: {key}")
                result[key] = value
            return result

        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_pairs)
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        if not {"target", "id_column"} <= payload.keys():
            raise ValueError("Configuration requires target and id_column")
        return cls(**payload)

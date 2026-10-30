import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import load_tabular, read_csv
from src.models import build_model
from src.train import run
from src.validation import holdout_indices


def make_data(path, task="regression"):
    path.mkdir()
    count = 60
    train = pd.DataFrame(
        {
            "id": [f"{i:03}" for i in range(count)],
            "number": np.arange(count, dtype=float),
            "category": ["a", "b", "c"] * 20,
            "target": np.arange(count) * 0.3 if task == "regression" else [0, 1] * 30,
        }
    )
    train.loc[2, "number"] = np.nan
    test = pd.DataFrame({"id": ["090", "091"], "number": [15.0, 25.0], "category": ["a", "new"]})
    sample = pd.DataFrame({"id": ["091", "090"], "target": [0.0, 0.0]})
    for name, frame in (("train", train), ("test", test), ("sample_submission", sample)):
        frame.to_csv(path / f"{name}.csv", index=False)
    return train


@pytest.mark.parametrize(
    "task,prediction",
    [("regression", "label"), ("classification", "label"), ("classification", "probability")],
)
def test_complete_runs_are_reproducible_and_align_ids(tmp_path, task, prediction):
    data = tmp_path / "data"
    make_data(data, task)
    config = Config(
        target="target",
        id_column="id",
        task=task,
        prediction=prediction,
        positive_label=1 if prediction == "probability" else None,
    )
    first = run(config, data, tmp_path / "first")
    second = run(config, data, tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/submission.csv").read_bytes() == (
        tmp_path / "second/submission.csv"
    ).read_bytes()
    submission = pd.read_csv(tmp_path / "first/submission.csv", dtype={"id": str})
    assert submission["id"].tolist() == ["091", "090"]
    if task == "regression":
        assert submission["target"].iloc[0] > submission["target"].iloc[1]
    if prediction == "probability":
        assert submission["target"].between(0, 1).all()
    assert json.loads((tmp_path / "first/status.json").read_text())["status"] == "complete"
    metadata = json.loads((tmp_path / "first/environment.json").read_text())
    assert len(metadata["inputs"]["train.csv"]) == 64
    assert metadata["features"] == ["number", "category"]
    with pytest.raises(FileExistsError):
        run(config, data, tmp_path / "first")


def test_preprocessing_never_fits_validation_rows():
    config = Config(target="target", id_column="id")
    frame = pd.DataFrame({"number": [1.0, 3.0, 10000.0], "category": ["a", "b", "secret"]})
    model = build_model(frame, config)
    model.fit(frame.iloc[:2], [1.0, 2.0])
    pre = model.named_steps["preprocess"]
    assert pre.named_transformers_["numeric"].named_steps["impute"].statistics_[0] == 2
    assert (
        "secret" not in pre.named_transformers_["categorical"].named_steps["encode"].categories_[0]
    )
    assert np.isfinite(model.predict(frame.iloc[2:])).all()


@pytest.mark.parametrize("split", ["group", "time"])
def test_split_has_no_group_or_timestamp_overlap(split):
    frame = pd.DataFrame(
        {
            "target": np.arange(20),
            "group": np.repeat(np.arange(5), 4),
            "time": pd.date_range("2025-01-01", periods=10).repeat(2).astype(str),
        }
    )
    config = Config(target="target", id_column="id", split=split, split_column=split)
    fit, valid = holdout_indices(frame, config)
    assert not set(fit) & set(valid)
    assert not set(frame.iloc[fit][split]) & set(frame.iloc[valid][split])
    if split == "time":
        assert frame.iloc[fit]["time"].max() < frame.iloc[valid]["time"].min()
    again = holdout_indices(frame, config)
    assert np.array_equal(fit, again[0])


def test_classification_split_rejects_missing_classes():
    frame = pd.DataFrame({"target": [0] * 10 + [1] * 10, "group": [0] * 10 + [1] * 10})
    config = Config(
        target="target", id_column="id", task="classification", split="group", split_column="group"
    )
    with pytest.raises(ValueError, match="every class"):
        holdout_indices(frame, config)


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"validation_size": 0},
        {"validation_size": float("nan")},
        {"task": "typo"},
        {"prediction": "probability"},
        {"split": "group"},
        {"id_column": "target"},
        {"split_column": "group"},
    ],
)
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        Config(**({"target": "target", "id_column": "id"} | change))


@pytest.mark.parametrize(
    "content",
    [
        '{"target":"x","id_column":"id","seeed":42}',
        '{"target":"x","target":"y","id_column":"id"}',
        "[]",
        "{}",
    ],
)
def test_config_load_rejects_typos_and_duplicate_keys(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        Config.load(path)


@pytest.mark.parametrize(
    "failure", ["schema", "duplicate_ids", "sample_ids", "missing_target", "inf"]
)
def test_data_validation(tmp_path, failure):
    data = tmp_path / "data"
    make_data(data)
    config = Config(target="target", id_column="id")
    train = pd.read_csv(data / "train.csv")
    if failure == "schema":
        train["extra"] = 1
    elif failure == "duplicate_ids":
        train.loc[1, "id"] = train.loc[0, "id"]
    elif failure == "sample_ids":
        sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": str})
        sample.loc[0, "id"] = "missing"
        sample.to_csv(data / "sample_submission.csv", index=False)
    elif failure == "missing_target":
        train.loc[0, "target"] = np.nan
    else:
        train.loc[0, "number"] = np.inf
    train.to_csv(data / "train.csv", index=False)
    with pytest.raises(ValueError):
        load_tabular(data, config)


def test_duplicate_csv_headers_rejected(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("id,x,x\na,1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_csv(path, id_column="id")


def test_failed_training_records_status(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_data(data)

    def fail(*args):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr("src.train.build_model", fail)
    with pytest.raises(RuntimeError):
        run(Config(target="target", id_column="id"), data, tmp_path / "failed")
    status = json.loads((tmp_path / "failed/status.json").read_text())
    assert status == {"status": "failed", "error_type": "RuntimeError"}
    assert not (tmp_path / "failed/submission.csv").exists()


def test_cli_smoke(tmp_path):
    data = tmp_path / "data"
    make_data(data)
    config = tmp_path / "config.json"
    config.write_text('{"target":"target","id_column":"id"}')
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config),
            "--data-dir",
            str(data),
            "--run-dir",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "rmse:" in result.stdout


@pytest.mark.parametrize("split", ["group", "time"])
def test_complete_group_and_time_runs(tmp_path, split):
    data = tmp_path / "data"
    make_data(data)
    train = pd.read_csv(data / "train.csv", dtype={"id": str})
    train[split] = (
        np.repeat(np.arange(15), 4)
        if split == "group"
        else pd.date_range("2025-01-01", periods=30).repeat(2).astype(str)
    )
    train.to_csv(data / "train.csv", index=False)
    config = Config(target="target", id_column="id", split=split, split_column=split)
    run(config, data, tmp_path / "run")
    metadata = json.loads((tmp_path / "run/environment.json").read_text())
    assert split not in metadata["features"]
    assert "train.py" in metadata["source_sha256"]


@pytest.mark.parametrize("mode", ["strings", "multiclass", "empty_features"])
def test_baseline_handles_edge_case_features_and_labels(tmp_path, mode):
    data = tmp_path / "data"
    make_data(data, "classification")
    train = pd.read_csv(data / "train.csv", dtype={"id": str})
    prediction, positive = "label", None
    if mode == "strings":
        train["target"] = train["target"].map({0: "no", 1: "yes"})
        prediction, positive = "probability", "yes"
    elif mode == "multiclass":
        train["target"] = [0, 1, 2] * 20
    else:
        train["category"] = np.nan
    train.to_csv(data / "train.csv", index=False)
    config = Config(
        target="target",
        id_column="id",
        task="classification",
        prediction=prediction,
        positive_label=positive,
    )
    run(config, data, tmp_path / "run")
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "complete"


def test_malformed_csv_row_rejected(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("id,x,target\na,1,2,extra\n")
    with pytest.raises(ValueError, match="fields"):
        read_csv(path, id_column="id")


def test_empty_columns_are_selected_using_only_fit_partition():
    config = Config(target="target", id_column="id")
    frame = pd.DataFrame({"number": [1.0, 3.0, 9.0], "only_validation": [np.nan, np.nan, 99.0]})
    model = build_model(frame, config)
    model.fit(frame.iloc[:2], [1.0, 3.0])
    numeric = model.named_steps["preprocess"].transformers_[0][2]
    assert numeric == ["number"]
    assert np.isfinite(model.predict(frame.iloc[2:])).all()

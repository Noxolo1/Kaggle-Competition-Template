from pathlib import Path

import pytest

from src.paths import find_data_root, project_root


def dataset(path):
    path.mkdir(parents=True)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (path / name).touch()
    return path


def test_explicit_and_environment_precedence(tmp_path, monkeypatch):
    explicit = dataset(tmp_path / "explicit")
    environment = dataset(tmp_path / "environment")
    monkeypatch.setenv("KAGGLE_DATA_DIR", str(environment))
    assert find_data_root(data_dir=explicit) == explicit
    assert find_data_root() == environment
    with pytest.raises(FileNotFoundError):
        find_data_root(data_dir=tmp_path / "missing")


def test_nested_project_and_local_discovery(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/paths.py").touch()
    nested = tmp_path / "notebooks/eda"
    nested.mkdir(parents=True)
    data = dataset(tmp_path / "data")
    monkeypatch.delenv("KAGGLE_DATA_DIR", raising=False)
    monkeypatch.chdir(nested)
    assert project_root() == tmp_path
    assert find_data_root() == data


@pytest.mark.parametrize("layout", ["competition", "competitions/competition"])
def test_kaggle_layouts(tmp_path, monkeypatch, layout):
    monkeypatch.delenv("KAGGLE_DATA_DIR", raising=False)
    data = dataset(tmp_path / "input" / layout)
    assert find_data_root("competition", kaggle_input=tmp_path / "input", root=tmp_path) == data


def test_ambiguous_roots_and_image_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_DATA_DIR", raising=False)
    dataset(tmp_path / "data")
    dataset(tmp_path / "input/competition")
    with pytest.raises(ValueError, match="Multiple"):
        find_data_root("competition", root=tmp_path, kaggle_input=tmp_path / "input")
    image = tmp_path / "images"
    (image / "train").mkdir(parents=True)
    (image / "test").mkdir()
    (image / "sample_submission.csv").touch()
    assert (
        find_data_root(data_dir=image, required=("train", "test", "sample_submission.csv")) == image
    )


def test_unsafe_required_path(tmp_path):
    with pytest.raises(ValueError):
        find_data_root(data_dir=tmp_path, required=("../secret",))


def test_repository_root():
    assert (project_root(Path(__file__)) / "pyproject.toml").is_file()

import numpy as np
import pandas as pd
import pytest

from src.submission import validate_submission, write_submission


@pytest.fixture
def sample():
    return pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]})


@pytest.mark.parametrize("failure", ["order", "rows", "schema", "missing", "inf", "probability"])
def test_rejects_invalid_submissions(sample, failure):
    frame = sample.copy()
    if failure == "order":
        frame = frame.iloc[::-1]
    elif failure == "rows":
        frame = frame.iloc[:1]
    elif failure == "schema":
        frame = frame[["target", "id"]]
    elif failure == "missing":
        frame.loc[0, "target"] = np.nan
    elif failure == "inf":
        frame.loc[0, "target"] = np.inf
    else:
        frame.loc[0, "target"] = 1.1
    with pytest.raises(ValueError):
        validate_submission(frame, sample, id_columns=["id"], probability_columns=["target"])


def test_write_no_index_no_overwrite_and_atomic_replace(tmp_path, sample):
    output = tmp_path / "nested/submission.csv"
    write_submission(sample, sample, output, id_columns=["id"])
    assert output.read_text().splitlines()[0] == "id,target"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_submission(sample, sample, output, id_columns=["id"])
    invalid = sample.assign(target=np.nan)
    with pytest.raises(ValueError):
        write_submission(invalid, sample, output, id_columns=["id"], overwrite=True)
    assert output.read_bytes() == original
    updated = sample.assign(target=[0.2, 0.7])
    write_submission(updated, sample, output, id_columns=["id"], overwrite=True)
    assert pd.read_csv(output)["target"].tolist() == [0.2, 0.7]


def test_string_labels_and_repeated_ids_are_supported():
    sample = pd.DataFrame({"id": ["a", "a"], "label": ["cat", "dog"]})
    validate_submission(sample.copy(), sample, id_columns=["id"])

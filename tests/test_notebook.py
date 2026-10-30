import ast

import nbformat

from src.paths import project_root


def test_notebook_is_valid_clean_and_compiles():
    notebook = nbformat.read(project_root() / "notebooks/template.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs
            ast.parse(cell.source)


def test_notebook_runs_from_nested_directory(tmp_path, monkeypatch):
    root = project_root()
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (data / name).touch()
    monkeypatch.setenv("KAGGLE_DATA_DIR", str(data))
    monkeypatch.delenv("KAGGLE_CODE_DIR", raising=False)
    monkeypatch.chdir(root / "notebooks")
    notebook = nbformat.read(root / "notebooks/template.ipynb", as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "<notebook>", "exec"), namespace)
    assert namespace["REPO_ROOT"] == root
    assert namespace["DATA_ROOT"] == data


def test_notebook_runs_with_explicit_attached_source(tmp_path, monkeypatch):
    root = project_root()
    monkeypatch.setenv("KAGGLE_CODE_DIR", str(root))
    monkeypatch.chdir(tmp_path)
    notebook = nbformat.read(root / "notebooks/template.ipynb", as_version=4)
    setup = next(cell for cell in notebook.cells if cell.cell_type == "code")
    namespace = {}
    exec(compile(setup.source, "<notebook>", "exec"), namespace)
    assert namespace["REPO_ROOT"] == root

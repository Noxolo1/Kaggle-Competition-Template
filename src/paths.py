"""Explicit, validated paths for local and Kaggle execution."""

import os
from collections.abc import Sequence
from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/paths.py").is_file():
            return candidate
    raise FileNotFoundError("Project root not found. Run inside the checkout or pass a root.")


def find_data_root(
    competition: str = "",
    *,
    data_dir: str | Path | None = None,
    root: str | Path | None = None,
    required: Sequence[str] = ("train.csv", "test.csv", "sample_submission.csv"),
    kaggle_input: str | Path = "/kaggle/input",
) -> Path:
    """Explicit argument > KAGGLE_DATA_DIR > matching mounts or local data/.

    Explicit paths fail immediately when incomplete. Automatic discovery rejects
    ambiguity. Required entries may be files or directories (e.g. train/ for images).
    """
    if not required or any(
        not name or Path(name).is_absolute() or ".." in Path(name).parts for name in required
    ):
        raise ValueError("required must contain relative paths without '..'")
    explicit = data_dir if data_dir is not None else os.environ.get("KAGGLE_DATA_DIR")
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        if competition and (
            competition in {".", ".."} or "/" in competition or "\\" in competition
        ):
            raise ValueError("competition must be a slug, not a path")
        candidates = []
        if competition:
            base = Path(kaggle_input)
            candidates += [base / competition, base / "competitions" / competition]
        try:
            candidates.append(Path(root).resolve() / "data" if root else project_root() / "data")
        except FileNotFoundError:
            pass
    matches = list(
        dict.fromkeys(
            path.resolve()
            for path in candidates
            if path.is_dir() and all((path / name).exists() for name in required)
        )
    )
    if len(matches) > 1:
        raise ValueError(f"Multiple data roots found: {matches}. Set data_dir explicitly.")
    if matches:
        return matches[0]
    checked = ", ".join(str(path) for path in candidates) or "no candidate directories"
    raise FileNotFoundError(f"Required {list(required)}; checked: {checked}")


def output_root(root: str | Path | None = None) -> Path:
    """Return an output location without creating directories as an import side effect."""
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working")
    return Path(root).resolve() if root else project_root()

# Kaggle Competition Template

A reusable local/Kaggle workspace with a runnable tabular baseline, validated paths
and submissions, reproducible experiment records, tests, and CI.

## Start a new competition

1. Create a repository from this template. Use Python 3.12 by default; CI covers
   Python 3.11â€“3.13 on Linux and Windows.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
   from the repository root:

   ```sh
   uv sync --locked --extra notebook --extra dev
   uv run --locked --extra notebook python -m ipykernel install --user --name kaggle-competition
   ```

   Select that kernel in your notebook editor. Give each competition its own
   kernel name. The committed `uv.lock` pins transitive dependencies.

   Without uv: create/activate a virtual environment, then run
   `python -m pip install -e ".[notebook,dev]"`.
   `python -m pip install -r requirements.txt` installs the core workflow only.
   Pip uses compatible version ranges rather than the uv lockfile.

3. Put private inputs in `data/`. Edit `configs/baseline.json` with the ID,
   target, task, competition slug, and validation strategy.
4. Run:

   ```sh
   uv run --locked python -m src.train --config configs/baseline.json --data-dir data
   ```

   Each run creates `runs/<UTC timestamp>-<suffix>/`. An explicit `--run-dir`
   must not exist. Inspect metrics, validation predictions, and the submission.
   The workflow never uploads to Kaggle automatically.

## Per-competition checklist

- Record competition URL, deadline, licenses, external-data rules, and inference limits.
- Record ID/target columns, missing-value conventions, dataset version, and official metric.
- Choose a validation strategy before feature engineering. Keep related entities
  together; use chronology for future predictions. Fit transforms, feature selection,
  target encoding, and resampling only on training partitions.
- Record baseline score, experiment changes, hardware, environment, and best run.
- Check final output semantics and competition rules before manually uploading.
  Keep a record of the run used for each submission.

Do not use the public leaderboard as your validation set.

## Baseline

Regression uses Ridge; classification uses logistic regression. Numeric imputation,
scaling, categorical imputation, and one-hot encoding live in the fitted pipeline.
Columns with no observed values in a fitting partition are ignored there.
The holdout score is measured before refitting the entire pipeline on all training rows.

| Config key | Purpose |
| --- | --- |
| `target`, `id_column` | Required training target and CSV identifier |
| `task` | `regression` (RMSE) or `classification` (accuracy) |
| `prediction` | `label`, or `probability` for binary classification (log loss) |
| `positive_label` | Required in probability mode; match the parsed target's type |
| `submission_target` | Output name when different from the training target |
| `split` | `random`, `group`, or `time` |
| `split_column` | Required for group/time; excluded from model features |
| `validation_size`, `seed` | Holdout fraction and random seed |
| `competition` | Slug for automatic Kaggle data discovery |

Unknown and duplicate config keys fail. Random classification splits are stratified.
Group splits keep entities together. Time splits parse UTC timestamps and keep all
tied timestamps in one partition, with training strictly earlier than validation.
Each classification partition must contain every class; incompatible splits fail.

Required files: `train.csv`, `test.csv`, `sample_submission.csv`. The baseline
requires one unique, nonempty ID per row and one prediction column. It rejects
schema mismatches, missing targets, infinite numeric inputs, and differing test/sample
IDs. Predictions are aligned to sample ID order, preserving sample column order.
IDs are read as strings to preserve leading zeros and literal `NA` IDs.
Blank CSV fields are missing; configure other missing markers in
`src/data_loader.py` for each competition.

This baseline loads CSVs in memory and one-hot encodes categoricals. Review
high-cardinality features and entity IDs before scaling it up. For images, text,
forecasting, multi-target outputs, repeated IDs, large data, custom metrics, or
cross-validation, use the notebook and helpers with your own modeling workflow.
Time tasks may need gap/purge periods to avoid overlapping-label leakage.

## Local and Kaggle paths

`src.paths.find_data_root` uses an explicit `data_dir`, then `KAGGLE_DATA_DIR`,
then checks the competition mounts and local `data/`:

- `/kaggle/input/<competition>/`
- `/kaggle/input/competitions/<competition>/`
- The checkout's `data/`, found from nested working directories.

Incomplete explicit paths fail. Multiple valid automatic candidates require an
explicit choice. Use `required=("train", "test", "sample_submission.csv")` for
image directories. Outputs use `/kaggle/working` on Kaggle and the project root locally.

On Kaggle, attach the repository source as a dataset and set `KAGGLE_CODE_DIR`
in the notebook's first cell to the directory containing `src/`. Set `DATA_DIR`
to the competition mount as needed. Check preinstalled library versions; Kaggle's
offline environment does not automatically match the local lockfile.

## Reproducibility and artifacts

Each completed run saves resolved config, Python/package versions, Git revision and dirty status
when available, source and input SHA-256 hashes, feature names, UTC start time, split row
positions, metric, validation predictions, and submission CSV. `status.json`
records running, complete, or failed. Failed runs retain diagnostic artifacts.

Commit code before important runs: a Git revision does not capture uncommitted
edits. Retain input files matching the recorded hashes. Seeds cover Python, NumPy,
and baseline estimators; exact results across library versions and hardware are
not guaranteed. Configure RNGs, workers, and deterministic algorithms separately
for optional frameworks.

`src.submission.write_submission` checks schema/order, row counts, ID values,
dtypes and order, missing values, non-finite numeric predictions, and declared
probability bounds. Pass explicit `id_columns` (or `[]` for ID-free formats).
It allows repeated IDs for custom workflows and refuses overwrites by default.
Check multi-class row sums, ranking rules, valid labels, and other competition
constraints separately.

## Optional packages

```sh
uv sync --locked --extra notebook --extra boosting --extra dev
uv sync --locked --extra kaggle
```

Pass all extras you want to retain on each sync; unselected extras are removed.
PyTorch is installed separately: use the
[official installation selector](https://pytorch.org/get-started/locally/) for
your OS, accelerator, and Python version. Follow its dedicated index command;
avoid adding a global extra package index. WSL is a Linux environment. Record
chosen torch/torchvision/torchaudio versions in the competition repository and
manage that environment explicitly; exact uv syncs can remove packages installed
outside the project.

## Verification and maintenance

```sh
uv sync --locked --extra dev
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
uv run --locked --extra dev pytest -W error
uv export --locked --no-dev --no-emit-project --no-hashes -o audit-requirements.txt
uv run --locked --extra dev pip-audit -r audit-requirements.txt
```

CI runs these checks without private data or credentials. It covers regression
tests, synthetic complete runs, notebook validity/cleanliness, and a core dependency
audit. Actions are pinned to verified commits with read-only repository permissions.
Dependabot proposes package and action updates. Audit optional dependencies when
enabled; the core audit does not include unselected extras. Remove the exported
audit file after use.

Upgrade deliberately with `uv lock --upgrade`, rerun checks, and commit the lockfile.
Keep credentials in environment variables or user-level Kaggle config. Data, models,
runs, submissions, virtual environments, and common credential files are ignored.
Ignore rules do not untrack existing files; review `git diff --cached` before pushing.
Never load untrusted pickle/joblib/torch checkpoints. Clear notebook outputs and
execution counts before committing; CI checks the starter notebook.

## Layout

```text
configs/baseline.json       Competition settings
data/                      Private inputs (ignored)
notebooks/template.ipynb    Local/Kaggle notebook
src/                       Shared paths, data, validation, training, submissions
tests/                     Synthetic regression and integration tests
runs/                      Per-run provenance and predictions (ignored)
models/                    Custom saved models (ignored)
submissions/               Custom submission exports (ignored)
pyproject.toml / uv.lock    Dependency definitions and resolved versions
```

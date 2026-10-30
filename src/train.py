"""Run a validation baseline: python -m src.train --config configs/baseline.json."""

import argparse
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, root_mean_squared_error

from src.config import Config
from src.data_loader import load_tabular
from src.models import build_model
from src.paths import find_data_root, output_root
from src.submission import write_submission
from src.utils import seed_everything, sha256_file, write_json
from src.validation import holdout_indices


def git_state() -> dict:
    try:
        root = Path(__file__).resolve().parent.parent

        def query(*args):
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()

        return {
            "revision": query("rev-parse", "HEAD"),
            "dirty": bool(query("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def run(config: Config, data_dir: Path, run_dir: Path) -> dict:
    """Create one exclusive run directory; failed runs retain a failure status for diagnosis."""
    seed_everything(config.seed)
    train, test, sample, features = load_tabular(data_dir, config)
    fit, valid = holdout_indices(train, config)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "status.json", {"status": "running"})
    try:
        write_json(run_dir / "config.json", asdict(config))
        write_json(
            run_dir / "environment.json",
            {
                "python": platform.python_version(),
                "packages": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "scipy",
                        "joblib",
                        "threadpoolctl",
                    )
                },
                "git": git_state(),
                "source_sha256": {
                    path.name: sha256_file(path)
                    for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
                },
                "started_utc": datetime.now(UTC).isoformat(),
                "inputs": {
                    name: sha256_file(data_dir / name)
                    for name in ("train.csv", "test.csv", "sample_submission.csv")
                },
                "features": features,
            },
        )
        write_json(
            run_dir / "split.json",
            {"train_row_positions": fit.tolist(), "validation_row_positions": valid.tolist()},
        )
        model = build_model(train[features], config)
        model.fit(train.iloc[fit][features], train.iloc[fit][config.target])
        validation = train.iloc[valid]
        if config.prediction == "probability":
            classes = model.classes_.tolist()
            positive_index = classes.index(config.positive_label)
            validation_predictions = model.predict_proba(validation[features])[:, positive_index]
            metric = "log_loss"
            score = log_loss(
                (validation[config.target] == config.positive_label).astype(int),
                validation_predictions,
                labels=[0, 1],
            )
        else:
            validation_predictions = model.predict(validation[features])
            metric = "rmse" if config.task == "regression" else "accuracy"
            scorer = root_mean_squared_error if config.task == "regression" else accuracy_score
            score = scorer(validation[config.target], validation_predictions)
        metrics = {"metric": metric, "score": float(score), "validation_rows": len(valid)}
        write_json(run_dir / "metrics.json", metrics)
        pd.DataFrame(
            {
                config.id_column: validation[config.id_column].to_numpy(),
                "actual": validation[config.target].to_numpy(),
                "prediction": validation_predictions,
            }
        ).to_csv(run_dir / "validation.csv", index=False)
        model.fit(train[features], train[config.target])
        predictions = (
            model.predict_proba(test[features])[
                :, model.classes_.tolist().index(config.positive_label)
            ]
            if config.prediction == "probability"
            else model.predict(test[features])
        )
        # Align by ID instead of assuming test and sample rows have the same order.
        aligned = pd.Series(predictions, index=test[config.id_column]).reindex(
            sample[config.id_column]
        )
        submission = sample.copy()
        submission[config.output_column] = aligned.to_numpy()
        write_submission(
            submission,
            sample,
            run_dir / "submission.csv",
            id_columns=[config.id_column],
            probability_columns=[config.output_column]
            if config.prediction == "probability"
            else [],
        )
        write_json(run_dir / "status.json", {"status": "complete"})
        return metrics
    except BaseException as error:
        write_json(
            run_dir / "status.json", {"status": "failed", "error_type": type(error).__name__}
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument(
        "--data-dir", type=Path, help="Overrides automatic discovery and KAGGLE_DATA_DIR"
    )
    parser.add_argument("--run-dir", type=Path, help="Must not already exist")
    args = parser.parse_args()
    config = Config.load(args.config)
    data_dir = find_data_root(config.competition, data_dir=args.data_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = args.run_dir or output_root() / "runs" / run_id
    metrics = run(config, data_dir, run_dir)
    print(f"{metrics['metric']}: {metrics['score']:.6f}")
    print(f"Run artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

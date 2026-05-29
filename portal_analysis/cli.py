"""CLI for training and evaluating hand movement classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from portal_analysis.config import MODELS_DIRECTORY
from portal_analysis.models.paths import ALL_TASKS, INFERENCE_TASKS, model_dir_for_task
from portal_analysis.training.pipeline import (
    evaluate_saved_model,
    predict_from_model,
    train_pipeline,
)
from portal_analysis.training.symptom_pipeline import train_all_symptom_models
from portal_analysis.training.task_config import TaskConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _REPO_ROOT / "configs"


def _config_path_for_task(task_name: str) -> Path:
    path = _CONFIGS_DIR / f"{task_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No config for task '{task_name}': {path}")
    return path


def cmd_train_symptoms(args: argparse.Namespace) -> None:
    tasks = list(INFERENCE_TASKS) if "all" in args.tasks else args.tasks
    version = args.version or "latest"

    for task_name in tasks:
        if task_name == "both_still":
            print(f"Skipping {task_name}: no symptom models.")
            continue
        config_path = (
            Path(args.config) if args.config and len(tasks) == 1 else _config_path_for_task(task_name)
        )
        task_config = TaskConfig.from_json(config_path)

        print(f"\n{'=' * 60}\nTraining symptoms: {task_config.task_name}\n{'=' * 60}")
        train_all_symptom_models(
            task_config=task_config,
            version=version,
            base_dir=args.processed_dir,
            models_root=args.models_dir,
            evaluate=not args.no_eval,
        )


def cmd_train(args: argparse.Namespace) -> None:
    tasks = list(ALL_TASKS) if "all" in args.tasks else args.tasks
    version = args.version or "latest"

    for task_name in tasks:
        config_path = Path(args.config) if args.config and len(tasks) == 1 else _config_path_for_task(task_name)
        task_config = TaskConfig.from_json(config_path)

        output_dir = (
            Path(args.output)
            if args.output and len(tasks) == 1
            else model_dir_for_task(task_config.task_name, version, args.models_dir)
        )

        print(f"\n{'=' * 60}\nTraining: {task_config.task_name}\n{'=' * 60}")
        train_pipeline(
            task_config=task_config,
            output_dir=output_dir,
            base_dir=args.processed_dir,
            version=version,
            evaluate=not args.no_eval,
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    evaluate_saved_model(Path(args.model), base_dir=args.processed_dir)


def cmd_predict(args: argparse.Namespace) -> None:
    if args.input is None:
        raise ValueError("Provide --input (.npy file).")

    X = np.load(args.input)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    y_pred = predict_from_model(Path(args.model), X)
    print("Predictions:", y_pred)

    if args.output:
        np.save(args.output, y_pred)
        print(f"Saved → {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PortalAnalysis — train and evaluate hand movement classifiers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train severity models")
    train_p.add_argument(
        "--tasks",
        nargs="+",
        choices=[*ALL_TASKS, "all"],
        default=["all"],
        help="Tasks to train (default: all)",
    )
    train_p.add_argument("--config", help="JSON config (single-task only)")
    train_p.add_argument("--output", help="Artifact output directory")
    train_p.add_argument("--version", default="latest", help="Model version tag")
    train_p.add_argument("--processed-dir", type=Path, default=None)
    train_p.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIRECTORY,
        help="Root directory for saved models",
    )
    train_p.add_argument("--no-eval", action="store_true")
    train_p.set_defaults(func=cmd_train)

    sym_p = sub.add_parser(
        "train-symptoms",
        help="Train binary classifiers for clinical motor signs (4 per task)",
    )
    sym_p.add_argument(
        "--tasks",
        nargs="+",
        choices=[*INFERENCE_TASKS, "all"],
        default=["all"],
        help="Tasks to train (default: all movement tasks)",
    )
    sym_p.add_argument("--config", help="JSON config (single-task only)")
    sym_p.add_argument("--version", default="latest", help="Model version tag")
    sym_p.add_argument("--processed-dir", type=Path, default=None)
    sym_p.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIRECTORY,
        help="Root directory for saved models",
    )
    sym_p.add_argument("--no-eval", action="store_true")
    sym_p.set_defaults(func=cmd_train_symptoms)

    eval_p = sub.add_parser("evaluate", help="Evaluate on held-out test set")
    eval_p.add_argument("--model", required=True, help="Artifact directory")
    eval_p.add_argument("--processed-dir", type=Path, default=None)
    eval_p.set_defaults(func=cmd_evaluate)

    pred_p = sub.add_parser("predict", help="Predict from .npy sequences")
    pred_p.add_argument("--model", required=True)
    pred_p.add_argument("--input", required=True)
    pred_p.add_argument("--output")
    pred_p.set_defaults(func=cmd_predict)

    return parser


def main(argv: list | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

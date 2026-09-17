"""Trains registered model(s) and serializes artifacts + benchmark metrics.

Usage:
    python scripts/train.py                        # forward, all registered models
    python scripts/train.py --models knn            # trains just knn
    python scripts/train.py --k 60                  # override dataset expansion value
    python scripts/train.py --direction reverse     # composition -> mechanical properties

Writes, per model, to artifacts/<model_name>/ (forward) or
artifacts/reverse/<model_name>/ (reverse):
    models.pkl    - dict of {target_column: fitted BaseCompositionModel}
    bounds.json   - input-feature min/max used for normalization
    metrics.json  - cross-validated benchmark (see src/models/evaluation.py)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import (
    DEFAULT_EXPANSION_K,
    MECHANICAL_PROPERTIES,
    TARGET_COLUMNS,
    build_reverse_training_dataset,
    build_training_dataset,
)
from src.models.evaluation import evaluate_model_on_targets
from src.models.registry import MODEL_REGISTRY

DATASET_PATH = ROOT / "data" / "raw" / "BS_EN_Dataset.xlsx"
ARTIFACTS_DIR = ROOT / "artifacts"


def train_and_save(model_name: str, direction: str = "forward", k: int = DEFAULT_EXPANSION_K) -> dict:
    if direction == "forward":
        dataset_builder = build_training_dataset
        target_columns = TARGET_COLUMNS
        out_dir = ARTIFACTS_DIR / model_name
    else:
        dataset_builder = build_reverse_training_dataset
        target_columns = MECHANICAL_PROPERTIES
        out_dir = ARTIFACTS_DIR / "reverse" / model_name

    print(f"[{direction}/{model_name}] building training dataset (k={k})...")
    expanded, features, bounds = dataset_builder(str(DATASET_PATH), k=k)
    targets = {col: expanded[col].to_numpy() for col in target_columns}

    model_cls = MODEL_REGISTRY[model_name]

    print(f"[{direction}/{model_name}] benchmarking via 5-fold cross-validation...")
    eval_results = evaluate_model_on_targets(model_cls(), features, targets, n_folds=5)

    print(f"[{direction}/{model_name}] fitting final models on the full dataset...")
    fitted_models = {}
    for col in target_columns:
        m = model_cls()
        m.fit(features, targets[col])
        fitted_models[col] = m

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "models.pkl", "wb") as f:
        pickle.dump(fitted_models, f)

    with open(out_dir / "bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)

    targets_metrics = {
        col: {
            "r2_mean": r.r2_mean,
            "r2_std": r.r2_std,
            "rmse_mean": r.rmse_mean,
            "rmse_std": r.rmse_std,
            "normalized_rmse_mean": r.normalized_rmse_mean,
        }
        for col, r in eval_results.items()
    }
    metrics = {
        "model_name": model_name,
        "direction": direction,
        "expansion_k": k,
        "n_training_rows": len(expanded),
        "targets": targets_metrics,
        "average_normalized_rmse": sum(
            v["normalized_rmse_mean"] for v in targets_metrics.values()
        ) / len(targets_metrics),
        "average_r2": sum(v["r2_mean"] for v in targets_metrics.values()) / len(targets_metrics),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[{direction}/{model_name}] done. avg normalized RMSE = {metrics['average_normalized_rmse']:.4f}, "
        f"avg R2 = {metrics['average_r2']:.4f}  -> {out_dir}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--k", type=int, default=DEFAULT_EXPANSION_K)
    args = parser.parse_args()

    for name in args.models:
        if name not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
        train_and_save(name, direction=args.direction, k=args.k)


if __name__ == "__main__":
    main()

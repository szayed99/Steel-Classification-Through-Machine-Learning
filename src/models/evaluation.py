"""Shared cross-validation benchmarking harness for composition models.

Any model implementing BaseCompositionModel is evaluated identically here,
so results are comparable across model types (AGENTS.md design decision #1:
models are pluggable and benchmarked through one shared harness). Evaluation
is per-target: one model instance is fit per (element, min/max) column,
matching the "30 independent regressions" design (decision #3).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .base import BaseCompositionModel, EvalResult


def evaluate_model(
    model: BaseCompositionModel,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> EvalResult:
    """Cross-validate `model` on a single target column y."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    r2_scores = []
    rmse_scores = []
    for train_idx, test_idx in kf.split(X):
        fold_model = model.clone()
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[test_idx])
        r2_scores.append(r2_score(y[test_idx], preds))
        rmse_scores.append(mean_squared_error(y[test_idx], preds) ** 0.5)

    y_range = float(y.max() - y.min())
    normalized_rmse = (float(np.mean(rmse_scores)) / y_range) if y_range > 0 else 0.0

    return EvalResult(
        r2_mean=float(np.mean(r2_scores)),
        r2_std=float(np.std(r2_scores)),
        rmse_mean=float(np.mean(rmse_scores)),
        rmse_std=float(np.std(rmse_scores)),
        normalized_rmse_mean=normalized_rmse,
    )


def evaluate_model_on_targets(
    model: BaseCompositionModel,
    X: np.ndarray,
    targets: Mapping[str, np.ndarray],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, EvalResult]:
    """Evaluate `model` independently on each named target column."""
    return {
        name: evaluate_model(model, X, y, n_folds=n_folds, random_state=random_state)
        for name, y in targets.items()
    }


def average_r2(results: Mapping[str, EvalResult]) -> float:
    return float(np.mean([r.r2_mean for r in results.values()]))

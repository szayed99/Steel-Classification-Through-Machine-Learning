"""Shared interface every composition-prediction model must implement.

A "model" here is a single-target regressor: it predicts one numeric column
(e.g. min_C or max_Mn) from the three normalized mechanical-property inputs.
The registry/training script fits one instance per target column.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    r2_mean: float
    r2_std: float
    rmse_mean: float
    rmse_std: float
    normalized_rmse_mean: float


class BaseCompositionModel(ABC):
    """Interface for a single-target regressor used in the model registry.

    Implementations must be deterministic given a fixed random_state, since
    the evaluation harness relies on repeatable cross-validation results.
    """

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseCompositionModel":
        """Fit the model on normalized features X and target values y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for normalized feature rows X."""

    @abstractmethod
    def clone(self) -> "BaseCompositionModel":
        """Return a fresh, unfitted instance with the same hyperparameters."""

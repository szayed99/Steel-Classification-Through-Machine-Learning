from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import BaseCompositionModel


class HistGradientBoostingCompositionModel(BaseCompositionModel):
    """Histogram-based gradient boosting - sklearn's fast, well-regularized
    boosted-tree implementation. Often the strongest out-of-the-box option
    for small-to-medium tabular regression problems."""

    name = "hist_gradient_boosting"

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: int | None = 6,
        l2_regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self._model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingCompositionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def clone(self) -> "HistGradientBoostingCompositionModel":
        return HistGradientBoostingCompositionModel(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

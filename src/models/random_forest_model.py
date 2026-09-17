from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseCompositionModel


class RandomForestCompositionModel(BaseCompositionModel):
    """Random forest: averages many decorrelated regression trees."""

    name = "random_forest"

    def __init__(self, n_estimators: int = 300, max_depth: int | None = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestCompositionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def clone(self) -> "RandomForestCompositionModel":
        return RandomForestCompositionModel(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
        )

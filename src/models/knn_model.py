from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .base import BaseCompositionModel


class KNNCompositionModel(BaseCompositionModel):
    """Baseline model: K-nearest-neighbors regression, k=6 per source notebook."""

    name = "knn"

    def __init__(self, n_neighbors: int = 6):
        self.n_neighbors = n_neighbors
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNCompositionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def clone(self) -> "KNNCompositionModel":
        return KNNCompositionModel(n_neighbors=self.n_neighbors)

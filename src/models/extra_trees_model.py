from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .base import BaseCompositionModel


class ExtraTreesCompositionModel(BaseCompositionModel):
    """Extremely randomized trees: like random forest but with randomized
    split thresholds too, which often reduces variance further on small
    tabular datasets like this one."""

    name = "extra_trees"

    def __init__(self, n_estimators: int = 300, max_depth: int | None = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesCompositionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def clone(self) -> "ExtraTreesCompositionModel":
        return ExtraTreesCompositionModel(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
        )

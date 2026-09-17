from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .base import BaseCompositionModel


class ZeroInflatedCompositionModel(BaseCompositionModel):
    """Two-stage model for zero-inflated / near-zero-variance targets.

    Several composition targets (e.g. min_Nb: 99.3% of rows are exactly 0,
    std=0.0013) are "not required for most grades, a real variable value for
    the rest". A single regressor on a target like that is fragile - tiny
    absolute errors near 0 explode R² because almost all of the target's
    variance comes from a handful of nonzero rows.

    This splits the problem: a classifier predicts whether the element is
    required at all (y == 0 vs y != 0), and a regressor - fit only on the
    nonzero subset - predicts the magnitude when it is. Degenerates cleanly
    to a plain regressor when a target has no exact-zero rows (e.g. the
    synthetic data in the shared model-contract test), and to a constant-zero
    predictor when a target is zero in every training row.
    """

    name = "zero_inflated"

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
        self._mode = None  # "all_zero" | "all_nonzero" | "mixed"
        self._classifier = None
        self._regressor = None

    def _make_classifier(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    def _make_regressor(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZeroInflatedCompositionModel":
        is_nonzero = y != 0.0

        if not is_nonzero.any():
            self._mode = "all_zero"
            return self

        if is_nonzero.all():
            self._mode = "all_nonzero"
            self._regressor = self._make_regressor()
            self._regressor.fit(X, y)
            return self

        self._mode = "mixed"
        self._classifier = self._make_classifier()
        self._classifier.fit(X, is_nonzero.astype(int))
        self._regressor = self._make_regressor()
        self._regressor.fit(X[is_nonzero], y[is_nonzero])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "all_zero":
            return np.zeros(len(X))

        if self._mode == "all_nonzero":
            return self._regressor.predict(X)

        preds = np.zeros(len(X))
        presence = self._classifier.predict(X).astype(bool)
        if presence.any():
            preds[presence] = self._regressor.predict(X[presence])
        return preds

    def clone(self) -> "ZeroInflatedCompositionModel":
        return ZeroInflatedCompositionModel(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

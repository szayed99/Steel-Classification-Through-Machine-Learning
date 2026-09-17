from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .base import BaseCompositionModel


class PolyRidgeCompositionModel(BaseCompositionModel):
    """Polynomial features + Ridge regression.

    A regularized version of the source notebook's polynomial-regression
    approach. The notebook used plain LinearRegression at degree=13, which
    is prone to overfitting at that degree (no regularization term at all);
    Ridge's L2 penalty keeps a moderate-degree polynomial fit stable.
    """

    name = "poly_ridge"

    def __init__(self, degree: int = 3, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha
        self._model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolyRidgeCompositionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def clone(self) -> "PolyRidgeCompositionModel":
        return PolyRidgeCompositionModel(degree=self.degree, alpha=self.alpha)

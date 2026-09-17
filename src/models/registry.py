"""Maps model names to classes, and declares which one is "active" by default.

Adding a new model (Random Forest, XGBoost, ...) means adding a file in this
package that implements BaseCompositionModel and registering it here — never
forking app.py or the training script. See AGENTS.md decision #1 and #4.
"""
from __future__ import annotations

from .base import BaseCompositionModel
from .extra_trees_model import ExtraTreesCompositionModel
from .hist_gradient_boosting_model import HistGradientBoostingCompositionModel
from .knn_model import KNNCompositionModel
from .poly_ridge_model import PolyRidgeCompositionModel
from .random_forest_model import RandomForestCompositionModel
from .zero_inflated_model import ZeroInflatedCompositionModel

MODEL_REGISTRY: dict[str, type[BaseCompositionModel]] = {
    "knn": KNNCompositionModel,
    "random_forest": RandomForestCompositionModel,
    "extra_trees": ExtraTreesCompositionModel,
    "hist_gradient_boosting": HistGradientBoostingCompositionModel,
    "poly_ridge": PolyRidgeCompositionModel,
    "zero_inflated": ZeroInflatedCompositionModel,
}

# The model used for production predictions by default. The UI's model
# selector can override this per-session; this is only the default.
#
# Promoted from "knn" in Phase 7 after benchmarking all registered models
# through the same evaluation harness (see PLAN.md Phase 7 for the full
# comparison table): hist_gradient_boosting had the lowest average
# normalized RMSE (0.0663 vs knn's 0.0720) and highest average R²
# (0.7869 vs 0.7551) of every candidate tried.
ACTIVE_MODEL = "hist_gradient_boosting"

# Default model for the reverse direction (composition -> mechanical
# properties, Phase 9). Benchmarked separately - the winner here isn't
# guaranteed to match ACTIVE_MODEL, and in fact the picture is different:
# random_forest, extra_trees, hist_gradient_boosting, and zero_inflated all
# tied EXACTLY (avg normalized RMSE 0.0416, avg R² 0.9286). This isn't a
# coincidence - composition is constant across all ~75 synthetic rows of a
# given grade (only mechanical properties vary within a grade), so this
# direction reduces to "identify the grade from its 30-dim composition
# fingerprint, then output that grade's mean property." Any sufficiently
# expressive model solves that near-perfectly and converges to the same
# ceiling (also why R² is much higher here than the forward direction's
# 0.7869 - a fundamentally easier task, not a better model). knn (0.0725,
# 0.7658) and poly_ridge (0.0573, 0.8866) couldn't carve the same sharp
# per-grade boundaries and trailed behind. Picked hist_gradient_boosting
# from the tied winners for continuity with ACTIVE_MODEL rather than
# introducing a second model family into the story.
ACTIVE_MODEL_REVERSE = "hist_gradient_boosting"


def get_model_class(name: str) -> type[BaseCompositionModel]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}") from exc


def create_model(name: str = ACTIVE_MODEL, **kwargs) -> BaseCompositionModel:
    return get_model_class(name)(**kwargs)

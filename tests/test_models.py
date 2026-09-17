import numpy as np
import pytest

from tests.conftest import DATASET_PATH, cached_reverse_training_dataset, cached_training_dataset
from src.data_pipeline import MECHANICAL_PROPERTIES, TARGET_COLUMNS, build_reverse_training_dataset
from src.models.base import BaseCompositionModel
from src.models.evaluation import evaluate_model, evaluate_model_on_targets
from src.models.knn_model import KNNCompositionModel
from src.models.registry import (
    ACTIVE_MODEL,
    ACTIVE_MODEL_REVERSE,
    MODEL_REGISTRY,
    create_model,
    get_model_class,
)

# Gated on normalized RMSE (RMSE / target range), not raw R². Several targets
# (e.g. min_Nb, min_Ti) are near-constant across the dataset - most grades
# specify no minimum for that element - which makes R² statistically
# unstable (tiny target variance amplifies noise, sometimes to negative R²)
# even when absolute prediction error is negligible. The source notebook hit
# the same issue and computed normalized RMSE for exactly this reason.
NORMALIZED_RMSE_FLOOR = 0.15


def assert_conforms_to_contract(model: BaseCompositionModel):
    """Shared conformance check any BaseCompositionModel implementation must pass."""
    rng = np.random.RandomState(0)
    X = rng.rand(50, 3)
    y = rng.rand(50)

    fitted = model.fit(X, y)
    assert fitted is not None

    preds = model.predict(X)
    assert preds.shape == (50,)

    clone = model.clone()
    assert clone is not model
    assert isinstance(clone, type(model))

    clone.fit(X, y)
    preds_2 = clone.predict(X)
    np.testing.assert_allclose(preds, preds_2, err_msg="same data + params must give same predictions")


class TestModelContract:
    def test_knn_conforms(self):
        assert_conforms_to_contract(KNNCompositionModel(n_neighbors=3))

    def test_all_registered_models_conform(self):
        for name, cls in MODEL_REGISTRY.items():
            assert_conforms_to_contract(cls())


class TestRegistry:
    def test_active_model_is_registered(self):
        assert ACTIVE_MODEL in MODEL_REGISTRY

    def test_active_model_reverse_is_registered(self):
        assert ACTIVE_MODEL_REVERSE in MODEL_REGISTRY

    def test_get_model_class_resolves(self):
        assert get_model_class("knn") is KNNCompositionModel

    def test_get_model_class_unknown_raises(self):
        with pytest.raises(ValueError):
            get_model_class("does-not-exist")

    def test_create_model_default_is_active(self):
        model = create_model()
        assert model.name == ACTIVE_MODEL


class TestEvaluationHarnessDeterminism:
    def test_same_seed_same_metrics(self):
        rng = np.random.RandomState(1)
        X = rng.rand(80, 3)
        y = rng.rand(80)
        model = KNNCompositionModel(n_neighbors=5)

        result_a = evaluate_model(model, X, y, n_folds=4, random_state=42)
        result_b = evaluate_model(model, X, y, n_folds=4, random_state=42)

        assert result_a.r2_mean == pytest.approx(result_b.r2_mean)
        assert result_a.rmse_mean == pytest.approx(result_b.rmse_mean)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not present")
class TestKNNBaselineBenchmark:
    @pytest.fixture(scope="class")
    def benchmark(self):
        expanded, features, _ = cached_training_dataset(k=75)
        targets = {col: expanded[col].to_numpy() for col in TARGET_COLUMNS}
        model = KNNCompositionModel(n_neighbors=6)
        results = evaluate_model_on_targets(model, features, targets, n_folds=5)
        return results

    def test_average_normalized_rmse_meets_floor(self, benchmark):
        avg_nrmse = float(np.mean([r.normalized_rmse_mean for r in benchmark.values()]))
        assert avg_nrmse <= NORMALIZED_RMSE_FLOOR

    def test_every_target_has_a_result(self, benchmark):
        assert set(benchmark.keys()) == set(TARGET_COLUMNS)

    def test_r2_recorded_for_visibility(self, benchmark):
        # Not gated (see NORMALIZED_RMSE_FLOOR comment above), but every
        # target must have a finite R² recorded for the Model Info UI tab.
        for result in benchmark.values():
            assert np.isfinite(result.r2_mean)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not present")
class TestReverseBenchmark:
    """Phase 9: composition -> mechanical properties. Benchmarks
    ACTIVE_MODEL_REVERSE through the identical harness, not KNN - the
    reverse direction's model comparison already picked the winner (see
    registry.py), unlike the forward direction which started from a KNN
    baseline and was upgraded later."""

    @pytest.fixture(scope="class")
    def benchmark(self):
        expanded, features, _ = cached_reverse_training_dataset(k=75)
        targets = {col: expanded[col].to_numpy() for col in MECHANICAL_PROPERTIES}
        model = get_model_class(ACTIVE_MODEL_REVERSE)()
        results = evaluate_model_on_targets(model, features, targets, n_folds=5)
        return results

    def test_average_normalized_rmse_meets_floor(self, benchmark):
        avg_nrmse = float(np.mean([r.normalized_rmse_mean for r in benchmark.values()]))
        assert avg_nrmse <= NORMALIZED_RMSE_FLOOR

    def test_every_property_has_a_result(self, benchmark):
        assert set(benchmark.keys()) == set(MECHANICAL_PROPERTIES)

    def test_r2_recorded_for_visibility(self, benchmark):
        for result in benchmark.values():
            assert np.isfinite(result.r2_mean)

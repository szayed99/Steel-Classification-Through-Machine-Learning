import subprocess
import sys
from pathlib import Path

import pytest

from src.data_pipeline import ELEMENTS, MECHANICAL_PROPERTIES
from src.models.predict import (
    ARTIFACTS_DIR,
    REVERSE_ARTIFACTS_DIR,
    PredictionService,
    ReversePredictionService,
    available_models,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def ensure_trained():
    """Phase 3 tests need real trained artifacts; train once if missing."""
    if not (ARTIFACTS_DIR / "knn" / "models.pkl").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--models", "knn"],
            check=True,
            cwd=ROOT,
        )
    if not (REVERSE_ARTIFACTS_DIR / "knn" / "models.pkl").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--models", "knn", "--direction", "reverse"],
            check=True,
            cwd=ROOT,
        )


@pytest.fixture(scope="module")
def service():
    return PredictionService("knn")


class TestAvailableModels:
    def test_knn_is_available_after_training(self):
        assert "knn" in available_models()


class TestPredictionService:
    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            PredictionService("does-not-exist")

    def test_predicts_all_elements(self, service):
        result = service.predict(16, 345, 580)
        assert set(result["composition"].keys()) == set(ELEMENTS)

    def test_min_never_exceeds_max(self, service):
        result = service.predict(16, 345, 580)
        for elem, bounds in result["composition"].items():
            assert bounds["min"] <= bounds["max"], f"{elem}: min > max"

    def test_no_negative_composition_values(self, service):
        result = service.predict(16, 345, 580)
        for bounds in result["composition"].values():
            assert bounds["min"] >= 0.0
            assert bounds["max"] >= 0.0

    def test_in_range_input_not_flagged(self, service):
        # 16/345/580 sits within the training data's observed ranges.
        result = service.predict(16, 345, 580)
        assert result["any_out_of_range"] is False

    def test_far_out_of_range_input_is_flagged(self, service):
        result = service.predict(elongation=1000, yield_strength=345, tensile_strength=580)
        assert result["out_of_range"]["Elongation"] is True
        assert result["any_out_of_range"] is True

    def test_deterministic_across_instances(self):
        # Loading the artifact fresh must reproduce the same prediction as
        # any other instance - no hidden state, no training-time randomness.
        a = PredictionService("knn").predict(16, 345, 580)
        b = PredictionService("knn").predict(16, 345, 580)
        assert a["composition"] == b["composition"]


@pytest.fixture(scope="module")
def reverse_service():
    return ReversePredictionService("knn")


# A typical, in-range composition (near the dataset's own medians for the
# "key" elements - see data_pipeline.py KEY_ELEMENTS).
_TYPICAL_COMPOSITION = {"C": 0.1, "Si": 0.2, "Mn": 0.8, "P": 0.015, "S": 0.013}


class TestAvailableReverseModels:
    def test_knn_is_available_after_training(self):
        from src.models.predict import available_reverse_models

        assert "knn" in available_reverse_models()


class TestReversePredictionService:
    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            ReversePredictionService("does-not-exist")

    def test_predicts_all_mechanical_properties(self, reverse_service):
        result = reverse_service.predict(_TYPICAL_COMPOSITION)
        assert set(result["properties"].keys()) == set(MECHANICAL_PROPERTIES)

    def test_no_negative_property_values(self, reverse_service):
        result = reverse_service.predict(_TYPICAL_COMPOSITION)
        for value in result["properties"].values():
            assert value >= 0.0

    def test_omitted_elements_default_to_zero(self, reverse_service):
        # Only specifying C should behave the same as explicitly zeroing
        # every other element - no KeyError, no special-casing needed.
        result_omitted = reverse_service.predict({"C": 0.1})
        result_explicit = reverse_service.predict({e: 0.0 for e in ELEMENTS} | {"C": 0.1})
        assert result_omitted["properties"] == result_explicit["properties"]

    def test_in_range_composition_not_flagged(self, reverse_service):
        result = reverse_service.predict(_TYPICAL_COMPOSITION)
        assert result["any_out_of_range"] is False

    def test_far_out_of_range_composition_is_flagged(self, reverse_service):
        # No real element in this dataset exceeds a few percent (see
        # data_pipeline.py's typo-correction threshold) - 50% is absurd.
        result = reverse_service.predict({"C": 50.0})
        assert result["out_of_range"]["C"] is True
        assert result["any_out_of_range"] is True

    def test_deterministic_across_instances(self):
        a = ReversePredictionService("knn").predict(_TYPICAL_COMPOSITION)
        b = ReversePredictionService("knn").predict(_TYPICAL_COMPOSITION)
        assert a["properties"] == b["properties"]

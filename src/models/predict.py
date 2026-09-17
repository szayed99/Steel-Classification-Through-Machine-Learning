"""Loads trained model artifacts and predicts composition for new inputs."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.data_pipeline import (
    ELEMENTS,
    MECHANICAL_PROPERTIES,
    TARGET_COLUMNS,
    normalize_point,
    out_of_range_flags,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


class PredictionService:
    """Loads a trained model's artifacts once, then predicts cheaply per-call."""

    def __init__(self, model_name: str, artifacts_dir: Path = ARTIFACTS_DIR):
        self.model_name = model_name
        out_dir = artifacts_dir / model_name
        models_path = out_dir / "models.pkl"
        bounds_path = out_dir / "bounds.json"
        metrics_path = out_dir / "metrics.json"
        if not models_path.exists():
            raise FileNotFoundError(
                f"No trained artifacts for model '{model_name}' at {out_dir}. "
                "Run scripts/train.py first."
            )
        with open(models_path, "rb") as f:
            self._models: dict = pickle.load(f)
        with open(bounds_path) as f:
            raw_bounds = json.load(f)
        self.bounds = {k: tuple(v) for k, v in raw_bounds.items()}
        with open(metrics_path) as f:
            self.metrics = json.load(f)

    def predict(self, elongation: float, yield_strength: float, tensile_strength: float) -> dict:
        point = [elongation, yield_strength, tensile_strength]
        norm_point = normalize_point(point, self.bounds, MECHANICAL_PROPERTIES)
        flags = out_of_range_flags(point, self.bounds, MECHANICAL_PROPERTIES)

        composition = {}
        for elem in ELEMENTS:
            min_pred = float(self._models[f"min_{elem}"].predict(norm_point)[0])
            max_pred = float(self._models[f"max_{elem}"].predict(norm_point)[0])
            # KNN can occasionally predict min > max for tight/near-constant
            # targets, and chemical composition can't be negative - clip both.
            lo, hi = sorted((min_pred, max_pred))
            composition[elem] = {
                "min": round(max(lo, 0.0), 4),
                "max": round(max(hi, 0.0), 4),
            }

        return {
            "composition": composition,
            "out_of_range": flags,
            "any_out_of_range": any(flags.values()),
        }


def available_models(artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    if not artifacts_dir.exists():
        return []
    return sorted(
        p.name for p in artifacts_dir.iterdir() if p.is_dir() and (p / "models.pkl").exists()
    )


REVERSE_ARTIFACTS_DIR = ARTIFACTS_DIR / "reverse"


class ReversePredictionService:
    """Loads a trained reverse-direction model's artifacts, then predicts
    mechanical properties from a chemical composition input.

    A composition input is a single value per element (not a min/max band -
    unlike a grade spec, a specific piece of steel has one actual
    composition), so it's expanded into the model's 30-column min_/max_
    feature space by using the same value for both the min_ and max_ column
    of each element - a zero-width band is the correct representation of
    "this element is at exactly this value."
    """

    def __init__(self, model_name: str, artifacts_dir: Path = REVERSE_ARTIFACTS_DIR):
        self.model_name = model_name
        out_dir = artifacts_dir / model_name
        models_path = out_dir / "models.pkl"
        bounds_path = out_dir / "bounds.json"
        metrics_path = out_dir / "metrics.json"
        if not models_path.exists():
            raise FileNotFoundError(
                f"No trained reverse-direction artifacts for model '{model_name}' at {out_dir}. "
                "Run `python scripts/train.py --direction reverse` first."
            )
        with open(models_path, "rb") as f:
            self._models: dict = pickle.load(f)
        with open(bounds_path) as f:
            raw_bounds = json.load(f)
        self.bounds = {k: tuple(v) for k, v in raw_bounds.items()}
        with open(metrics_path) as f:
            self.metrics = json.load(f)

    def predict(self, composition: dict[str, float]) -> dict:
        """composition: {element_symbol: value (%)}. Elements omitted are
        treated as 0 (not specified) - the same convention the dataset
        itself uses for "no requirement"."""
        point = []
        for elem in ELEMENTS:
            value = float(composition.get(elem, 0.0))
            point.append(value)  # min_<elem>
            point.append(value)  # max_<elem>

        norm_point = normalize_point(point, self.bounds, TARGET_COLUMNS)
        # Deliberately NOT out_of_range_flags() against all 30 columns: a
        # min_<elem> column's observed range is "how low a grade's spec
        # floor gets" (typically narrow/low), not "how much of this element
        # a real sample can contain" - checking an actual composition value
        # against it produces false positives (e.g. min_Si tops out at 0.15
        # across every grade's spec floor, but 0.2% actual Si is completely
        # ordinary - within max_Si's real range of 0-0.6). The max_<elem>
        # column's upper bound is the dataset's true ceiling for that
        # element, so that's the only bound an actual value should be
        # checked against.
        element_flags = {
            elem: bool(composition.get(elem, 0.0) > self.bounds[f"max_{elem}"][1]) for elem in ELEMENTS
        }

        properties = {}
        for prop in MECHANICAL_PROPERTIES:
            pred = float(self._models[prop].predict(norm_point)[0])
            properties[prop] = round(max(pred, 0.0), 2)

        return {
            "properties": properties,
            "out_of_range": element_flags,
            "any_out_of_range": any(element_flags.values()),
        }


def available_reverse_models(artifacts_dir: Path = REVERSE_ARTIFACTS_DIR) -> list[str]:
    return available_models(artifacts_dir)

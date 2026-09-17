"""Load and preprocess the BS EN steel dataset.

The raw spreadsheet describes each steel grade as a spec *band* rather than
a single data point: chemical elements are given as "min-max" range strings,
and mechanical properties are sometimes single values, sometimes ranges.
This module turns that into a flat, normalized dataset of individual points
suitable for training one regressor per (element, min/max) target.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

MECHANICAL_PROPERTIES = ["Elongation", "Yield Strength", "Tensile Strength"]
ELEMENTS = ["C", "Si", "Mn", "P", "S", "N", "Cu", "Cr", "Ni", "Mo", "Al", "Ti", "V", "Nb", "B"]
GRADE_COLUMN = "Grade"
CLASSIFICATION_COLUMN = "Classification"
TARGET_COLUMNS = [f"{bound}{elem}" for elem in ELEMENTS for bound in ("min_", "max_")]

# Split used by the reverse-direction (composition -> properties) input form.
# Data-driven, not arbitrary: measured directly against the dataset - C, Si,
# Mn, P, S each have a max_<elem> zero-rate under 10% (specified for nearly
# every grade), while every other element is 35-95% zero (only specified for
# some alloy/special grades). See PLAN.md Phase 9 for the measured rates.
KEY_ELEMENTS = ["C", "Si", "Mn", "P", "S"]
TRACE_ELEMENTS = [e for e in ELEMENTS if e not in KEY_ELEMENTS]

# Found via sweep in the source notebook (optimal was ~73 for KNN, ~77 for
# polynomial regression); this sits in the middle of that range.
DEFAULT_EXPANSION_K = 75

# No chemical element in this dataset legitimately exceeds a few percent.
# A handful of source cells are missing a decimal point (e.g. S: "0-025"
# meant "0-0.025"); any parsed value above this is that typo, corrected by
# /1000 to match the exact corruption pattern observed in the raw file.
_ELEMENT_TYPO_THRESHOLD = 5.0


def load_raw_dataset(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def parse_range(raw) -> tuple[float, float]:
    """Parse a spec cell into (min, max) floats.

    Handles plain numbers, "-" meaning "no requirement" (-> 0.0, 0.0), a
    single value (min == max), whitespace around the separator, and noisy
    multi-dash cells (takes the first token as min, the last as max).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0.0, 0.0
    if isinstance(raw, (int, float)):
        return float(raw), float(raw)

    text = str(raw).strip()
    if text in ("", "-", "nan"):
        return 0.0, 0.0

    parts = [p.strip() for p in text.split("-") if p.strip() != ""]
    if not parts:
        return 0.0, 0.0
    if len(parts) == 1:
        value = float(parts[0])
        return value, value
    return float(parts[0]), float(parts[-1])


def split_element_ranges(df: pd.DataFrame, elements: Iterable[str] = ELEMENTS) -> pd.DataFrame:
    """Add min_<elem>/max_<elem> columns parsed from each element's range cell."""
    df = df.copy()
    elements = list(elements)
    for elem in elements:
        parsed = df[elem].map(parse_range)
        df[f"min_{elem}"] = parsed.map(lambda t: t[0])
        df[f"max_{elem}"] = parsed.map(lambda t: t[1])
    return _fix_element_decimal_typos(df, elements)


def _fix_element_decimal_typos(df: pd.DataFrame, elements: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for elem in elements:
        for bound in ("min_", "max_"):
            col = f"{bound}{elem}"
            mask = df[col] > _ELEMENT_TYPO_THRESHOLD
            df.loc[mask, col] = df.loc[mask, col] / 1000.0
    return df


def _expand_row(row: pd.Series, k: int) -> pd.DataFrame:
    ys_is_range = isinstance(row["Yield Strength"], str)
    ts_is_range = isinstance(row["Tensile Strength"], str)

    base = row.copy()
    if ys_is_range and ts_is_range:
        lo, hi = parse_range(row["Yield Strength"])
        base["Yield Strength"] = int(round((lo + hi) / 2))
        feature = "Tensile Strength"
        feat_values = _interpolate(row["Tensile Strength"], k)
    elif ts_is_range:
        feature = "Tensile Strength"
        feat_values = _interpolate(row["Tensile Strength"], k)
    elif ys_is_range:
        feature = "Yield Strength"
        feat_values = _interpolate(row["Yield Strength"], k)
    else:
        # Neither is a range: jitter Tensile Strength +/-4 so every grade
        # still contributes k rows and the expanded dataset stays balanced.
        ts = float(row["Tensile Strength"])
        feat_values = [int(round(v)) for v in np.linspace(ts - 4, ts + 5, k)]
        feature = "Tensile Strength"

    expanded = pd.DataFrame([base] * k).reset_index(drop=True)
    expanded[feature] = feat_values
    return expanded


def _interpolate(text: str, k: int) -> list[int]:
    lo, hi = parse_range(text)
    return [int(round(v)) for v in np.linspace(lo, hi, k)]


def expand_dataset(df: pd.DataFrame, k: int = DEFAULT_EXPANSION_K) -> pd.DataFrame:
    """Expand each grade's mechanical-property range(s) into k interpolated rows."""
    expanded = pd.concat(
        [_expand_row(row, k) for _, row in df.iterrows()],
        ignore_index=True,
    )
    return expanded.astype(
        {"Elongation": int, "Yield Strength": int, "Tensile Strength": int}
    )


def normalize_features(
    df: pd.DataFrame, columns: Iterable[str] = MECHANICAL_PROPERTIES
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Min-max normalize the mechanical-property columns.

    Returns the normalized feature matrix and the {column: (min, max)}
    bounds used, so the same bounds can be applied to new single-point
    inputs at inference time via normalize_point.
    """
    columns = list(columns)
    bounds = {col: (float(df[col].min()), float(df[col].max())) for col in columns}
    scaled = np.zeros((len(df), len(columns)), dtype=float)
    for i, col in enumerate(columns):
        lo, hi = bounds[col]
        span = hi - lo
        scaled[:, i] = 0.0 if span == 0 else (df[col].to_numpy() - lo) / span
    return scaled, bounds


def normalize_point(
    point: Iterable[float],
    bounds: dict[str, tuple[float, float]],
    columns: Iterable[str] = MECHANICAL_PROPERTIES,
) -> np.ndarray:
    """Apply previously-computed training bounds to a single new input point."""
    values = []
    for col, v in zip(columns, point):
        lo, hi = bounds[col]
        span = hi - lo
        values.append(0.0 if span == 0 else (v - lo) / span)
    return np.array([values])


def out_of_range_flags(
    point: Iterable[float],
    bounds: dict[str, tuple[float, float]],
    columns: Iterable[str] = MECHANICAL_PROPERTIES,
) -> dict[str, bool]:
    """Flag which inputs fall outside the training data's observed range."""
    return {col: bool(v < bounds[col][0] or v > bounds[col][1]) for col, v in zip(columns, point)}


def build_training_dataset(
    path: str, k: int = DEFAULT_EXPANSION_K
) -> tuple[pd.DataFrame, np.ndarray, dict[str, tuple[float, float]]]:
    """Load the raw dataset and produce the expanded, normalized training data.

    Returns (expanded_df, normalized_features, feature_bounds). Targets are
    read directly off expanded_df[TARGET_COLUMNS] by the training script.
    """
    raw = load_raw_dataset(path)
    split = split_element_ranges(raw)
    expanded = expand_dataset(split, k)
    features, bounds = normalize_features(expanded)
    return expanded, features, bounds


def build_reverse_training_dataset(
    path: str, k: int = DEFAULT_EXPANSION_K
) -> tuple[pd.DataFrame, np.ndarray, dict[str, tuple[float, float]]]:
    """Same pipeline as build_training_dataset, mirrored for the reverse
    direction: chemical composition (the 30 min_/max_ element columns) as
    input features, mechanical properties as what gets predicted. Reuses
    every step of the forward pipeline unchanged - only which columns get
    normalized as features differs.

    Returns (expanded_df, normalized_features, feature_bounds). Targets are
    read directly off expanded_df[MECHANICAL_PROPERTIES] by the training
    script.
    """
    raw = load_raw_dataset(path)
    split = split_element_ranges(raw)
    expanded = expand_dataset(split, k)
    features, bounds = normalize_features(expanded, columns=TARGET_COLUMNS)
    return expanded, features, bounds

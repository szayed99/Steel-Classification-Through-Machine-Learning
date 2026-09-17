"""Finds the real steel grades whose chemical composition (or mechanical
properties, for the reverse direction) most closely matches a prediction,
for interpretability alongside raw numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_pipeline import (
    ELEMENTS,
    GRADE_COLUMN,
    MECHANICAL_PROPERTIES,
    TARGET_COLUMNS,
    load_raw_dataset,
    parse_range,
    split_element_ranges,
)


def build_grade_reference(path: str) -> pd.DataFrame:
    """One row per real grade with its min_/max_ element columns - the table
    grade matching searches against."""
    raw = load_raw_dataset(path)
    return split_element_ranges(raw)


def find_similar_grades(
    predicted_composition: dict[str, dict[str, float]],
    reference: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """Return the top_n real grades closest to the predicted composition,
    ranked by ascending Euclidean distance across all min_/max_ element
    values.
    """
    predicted_vector = np.array(
        [predicted_composition[elem][bound] for elem in ELEMENTS for bound in ("min", "max")]
    )
    target_matrix = reference[TARGET_COLUMNS].to_numpy()
    distances = np.sqrt(((target_matrix - predicted_vector) ** 2).sum(axis=1))

    ranked = reference.assign(_distance=distances).sort_values("_distance")
    ranked = ranked.drop_duplicates(subset=GRADE_COLUMN)

    top = ranked.head(top_n)
    return [
        {"grade": row[GRADE_COLUMN], "distance": float(row["_distance"])}
        for _, row in top.iterrows()
    ]


def _mechanical_midpoint(value) -> float:
    """A grade's raw Elongation/Yield/Tensile Strength cell may be a single
    value or a "min-max" range string; collapse either to one number for
    distance calculations."""
    lo, hi = parse_range(value)
    return (lo + hi) / 2


def find_similar_grades_by_properties(
    predicted_properties: dict[str, float],
    reference: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """Reverse-direction counterpart to find_similar_grades: given predicted
    mechanical properties, return the top_n real grades whose actual
    properties are closest, ranked by ascending Euclidean distance.

    Elongation/Yield Strength/Tensile Strength have very different scales
    (roughly 0-40 / 200-1000 / 300-1200), so - unlike the composition match,
    where every value is a comparable small percentage - each property is
    min-max normalized against the reference set's own range before
    computing distance, so no single property dominates just by having
    larger raw numbers.
    """
    mech_matrix = np.array(
        [[_mechanical_midpoint(row[prop]) for prop in MECHANICAL_PROPERTIES] for _, row in reference.iterrows()]
    )
    lo = mech_matrix.min(axis=0)
    hi = mech_matrix.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)

    predicted_vector = np.array([predicted_properties[prop] for prop in MECHANICAL_PROPERTIES])
    norm_matrix = (mech_matrix - lo) / span
    norm_predicted = (predicted_vector - lo) / span

    distances = np.sqrt(((norm_matrix - norm_predicted) ** 2).sum(axis=1))

    ranked = reference.assign(_distance=distances).sort_values("_distance")
    ranked = ranked.drop_duplicates(subset=GRADE_COLUMN)

    top = ranked.head(top_n)
    return [
        {"grade": row[GRADE_COLUMN], "distance": float(row["_distance"])}
        for _, row in top.iterrows()
    ]

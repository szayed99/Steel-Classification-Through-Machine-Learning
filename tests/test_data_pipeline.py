import numpy as np
import pandas as pd
import pytest

from tests.conftest import DATASET_PATH
from src.data_pipeline import (
    ELEMENTS,
    KEY_ELEMENTS,
    MECHANICAL_PROPERTIES,
    TARGET_COLUMNS,
    TRACE_ELEMENTS,
    build_reverse_training_dataset,
    build_training_dataset,
    expand_dataset,
    load_raw_dataset,
    normalize_features,
    normalize_point,
    out_of_range_flags,
    parse_range,
    split_element_ranges,
)


class TestParseRange:
    def test_simple_range(self):
        assert parse_range("0.10-0.20") == (0.10, 0.20)

    def test_single_value_no_dash(self):
        assert parse_range("0.01") == (0.01, 0.01)

    def test_dash_only_means_no_requirement(self):
        assert parse_range("-") == (0.0, 0.0)

    def test_missing_value_means_no_requirement(self):
        assert parse_range(None) == (0.0, 0.0)
        assert parse_range(float("nan")) == (0.0, 0.0)

    def test_whitespace_around_separator(self):
        assert parse_range(" 0.3 - 0.8 ") == (0.3, 0.8)
        assert parse_range("0- 0.35") == (0.0, 0.35)

    def test_numeric_input_passthrough(self):
        assert parse_range(22) == (22.0, 22.0)

    def test_noisy_multi_dash_takes_first_and_last(self):
        # Observed in the raw dataset's Mn column: "0-0.3 - 0.8 "
        assert parse_range("0-0.3 - 0.8 ") == (0.0, 0.8)


class TestSplitElementRanges:
    def _sample_df(self):
        return pd.DataFrame(
            {
                "C": ["0.10-0.20"],
                "Si": ["0-0.4"],
                "Mn": ["0.5-1.5"],
                "P": ["0-0.03"],
                "S": ["0-025"],  # known dropped-decimal typo in source data
                "N": ["0-012"],  # same typo pattern
                "Cu": ["-"],
                "Cr": ["0-0.3"],
                "Ni": ["0-0.3"],
                "Mo": ["0-0.1"],
                "Al": ["0-0.2"],
                "Ti": ["0-0.05"],
                "V": ["0-0.05"],
                "Nb": ["0-0.05"],
                "B": ["-"],
            }
        )

    def test_creates_min_max_columns(self):
        df = split_element_ranges(self._sample_df())
        assert df.loc[0, "min_C"] == pytest.approx(0.10)
        assert df.loc[0, "max_C"] == pytest.approx(0.20)

    def test_dash_only_element_is_zero(self):
        df = split_element_ranges(self._sample_df())
        assert df.loc[0, "min_Cu"] == 0.0
        assert df.loc[0, "max_Cu"] == 0.0

    def test_corrects_dropped_decimal_typo(self):
        df = split_element_ranges(self._sample_df())
        # "0-025" must become 0-0.025, not 0-25 (25% sulfur is impossible)
        assert df.loc[0, "max_S"] == pytest.approx(0.025)
        assert df.loc[0, "max_N"] == pytest.approx(0.012)

    def test_all_target_columns_present(self):
        df = split_element_ranges(self._sample_df())
        for col in TARGET_COLUMNS:
            assert col in df.columns


class TestExpandDataset:
    def _row_with_range(self, k=10):
        df = pd.DataFrame(
            [
                {
                    "Grade": "TEST1",
                    "Elongation": 20,
                    "Yield Strength": 300,
                    "Tensile Strength": "400-500",
                }
            ]
        )
        return expand_dataset(df, k=k)

    def test_produces_exactly_k_rows(self):
        expanded = self._row_with_range(k=10)
        assert len(expanded) == 10

    def test_interpolates_across_the_full_range(self):
        expanded = self._row_with_range(k=10)
        values = sorted(expanded["Tensile Strength"].tolist())
        assert values[0] == 400
        assert values[-1] == 500

    def test_monotonic_interpolation(self):
        expanded = self._row_with_range(k=10)
        values = expanded["Tensile Strength"].tolist()
        assert values == sorted(values)

    def test_output_dtypes_are_int(self):
        expanded = self._row_with_range(k=5)
        for col in ["Elongation", "Yield Strength", "Tensile Strength"]:
            assert pd.api.types.is_integer_dtype(expanded[col])

    def test_non_range_row_still_expands_to_k_rows(self):
        df = pd.DataFrame(
            [{"Grade": "TEST2", "Elongation": 15, "Yield Strength": 250, "Tensile Strength": 450}]
        )
        expanded = expand_dataset(df, k=8)
        assert len(expanded) == 8
        assert expanded["Tensile Strength"].nunique() > 1  # jittered, not identical


class TestNormalize:
    def _df(self):
        return pd.DataFrame(
            {
                "Elongation": [10, 20, 30],
                "Yield Strength": [100, 200, 300],
                "Tensile Strength": [400, 500, 600],
            }
        )

    def test_output_within_unit_range(self):
        features, _ = normalize_features(self._df())
        assert features.min() >= 0.0
        assert features.max() <= 1.0

    def test_min_and_max_map_to_zero_and_one(self):
        features, bounds = normalize_features(self._df())
        assert features[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
        assert features[-1].tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_normalize_point_uses_training_bounds(self):
        _, bounds = normalize_features(self._df())
        point = normalize_point([10, 100, 400], bounds)
        assert point.tolist() == [[0.0, 0.0, 0.0]]
        point_mid = normalize_point([20, 200, 500], bounds)
        assert point_mid.tolist()[0] == pytest.approx([0.5, 0.5, 0.5])

    def test_out_of_range_flags(self):
        _, bounds = normalize_features(self._df())
        flags = out_of_range_flags([5, 200, 400], bounds)
        assert flags["Elongation"] is True
        assert flags["Yield Strength"] is False


class TestBuildTrainingDatasetIntegration:
    @pytest.fixture(scope="class")
    def built(self):
        if not DATASET_PATH.exists():
            pytest.skip("dataset not present")
        return build_training_dataset(str(DATASET_PATH), k=10)

    def test_row_count_matches_expansion(self, built):
        raw = load_raw_dataset(str(DATASET_PATH))
        expanded, features, bounds = built
        assert len(expanded) == len(raw) * 10

    def test_all_targets_present_and_finite(self, built):
        expanded, _, _ = built
        for col in TARGET_COLUMNS:
            assert col in expanded.columns
            assert np.isfinite(expanded[col]).all()

    def test_feature_matrix_shape(self, built):
        expanded, features, _ = built
        assert features.shape == (len(expanded), 3)

    def test_no_element_value_implausibly_large(self, built):
        # Catches unfixed decimal-typo-style corruption: no real element
        # composition in this dataset should exceed 5%.
        expanded, _, _ = built
        for elem in ELEMENTS:
            assert expanded[f"max_{elem}"].max() < 5.0


class TestKeyTraceElementSplit:
    def test_every_element_classified_exactly_once(self):
        assert sorted(KEY_ELEMENTS + TRACE_ELEMENTS) == sorted(ELEMENTS)
        assert set(KEY_ELEMENTS) & set(TRACE_ELEMENTS) == set()

    def test_key_elements_are_the_expected_five(self):
        # Data-driven claim (see data_pipeline.py comment): these five have
        # a max_<elem> zero-rate under 10% in the real dataset - specified
        # for nearly every grade - unlike every other element.
        assert set(KEY_ELEMENTS) == {"C", "Si", "Mn", "P", "S"}


class TestBuildReverseTrainingDatasetIntegration:
    @pytest.fixture(scope="class")
    def built(self):
        if not DATASET_PATH.exists():
            pytest.skip("dataset not present")
        return build_reverse_training_dataset(str(DATASET_PATH), k=10)

    def test_row_count_matches_expansion(self, built):
        raw = load_raw_dataset(str(DATASET_PATH))
        expanded, features, bounds = built
        assert len(expanded) == len(raw) * 10

    def test_feature_matrix_shape_is_30_composition_columns(self, built):
        expanded, features, _ = built
        assert features.shape == (len(expanded), len(TARGET_COLUMNS))

    def test_bounds_cover_composition_columns_not_mechanical(self, built):
        _, _, bounds = built
        assert set(bounds.keys()) == set(TARGET_COLUMNS)

    def test_mechanical_targets_present_and_finite(self, built):
        expanded, _, _ = built
        for col in MECHANICAL_PROPERTIES:
            assert col in expanded.columns
            assert np.isfinite(expanded[col]).all()

import pytest

from tests.conftest import DATASET_PATH
from src.data_pipeline import ELEMENTS, GRADE_COLUMN, MECHANICAL_PROPERTIES, parse_range
from src.grade_matcher import build_grade_reference, find_similar_grades, find_similar_grades_by_properties


@pytest.fixture(scope="module")
def reference():
    if not DATASET_PATH.exists():
        pytest.skip("dataset not present")
    return build_grade_reference(str(DATASET_PATH))


class TestFindSimilarGrades:
    def test_exact_composition_returns_that_grade_first(self, reference):
        row = reference.iloc[10]
        predicted = {
            elem: {"min": row[f"min_{elem}"], "max": row[f"max_{elem}"]} for elem in ELEMENTS
        }
        results = find_similar_grades(predicted, reference, top_n=3)
        assert results[0]["grade"] == row[GRADE_COLUMN]
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-9)

    def test_returns_requested_count_of_distinct_grades(self, reference):
        row = reference.iloc[0]
        predicted = {
            elem: {"min": row[f"min_{elem}"], "max": row[f"max_{elem}"]} for elem in ELEMENTS
        }
        results = find_similar_grades(predicted, reference, top_n=3)
        assert len(results) == 3
        grades = [r["grade"] for r in results]
        assert len(set(grades)) == 3

    def test_ranked_by_ascending_distance(self, reference):
        row = reference.iloc[0]
        predicted = {
            elem: {"min": row[f"min_{elem}"], "max": row[f"max_{elem}"]} for elem in ELEMENTS
        }
        results = find_similar_grades(predicted, reference, top_n=5)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)


class TestFindSimilarGradesByProperties:
    def _row_midpoints(self, row):
        return {prop: sum(parse_range(row[prop])) / 2 for prop in MECHANICAL_PROPERTIES}

    def test_exact_properties_returns_that_grade_first(self, reference):
        row = reference.iloc[10]
        predicted = self._row_midpoints(row)
        results = find_similar_grades_by_properties(predicted, reference, top_n=3)
        assert results[0]["grade"] == row[GRADE_COLUMN]
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-9)

    def test_returns_requested_count_of_distinct_grades(self, reference):
        row = reference.iloc[0]
        predicted = self._row_midpoints(row)
        results = find_similar_grades_by_properties(predicted, reference, top_n=3)
        assert len(results) == 3
        grades = [r["grade"] for r in results]
        assert len(set(grades)) == 3

    def test_ranked_by_ascending_distance(self, reference):
        row = reference.iloc[0]
        predicted = self._row_midpoints(row)
        results = find_similar_grades_by_properties(predicted, reference, top_n=5)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

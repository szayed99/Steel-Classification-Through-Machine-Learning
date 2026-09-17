import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import DATASET_PATH, ROOT
from src.models.predict import ARTIFACTS_DIR


@pytest.fixture(scope="module", autouse=True)
def ensure_trained():
    if not DATASET_PATH.exists():
        pytest.skip("dataset not present")
    if not (ARTIFACTS_DIR / "knn" / "models.pkl").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--models", "knn"],
            check=True,
            cwd=ROOT,
        )


def _run_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "src" / "app.py"))
    at.run(timeout=30)
    return at


class TestAppSmoke:
    def test_app_loads_without_exception(self):
        at = _run_app()
        assert not at.exception

    def test_golden_path_renders_results(self):
        at = _run_app()
        # defaults (20 / 300 / 500) are within training range
        at.button[0].click().run(timeout=30)
        assert not at.exception
        assert len(at.tabs) == 3
        assert len(at.dataframe) > 0

    def test_out_of_range_input_shows_warning(self):
        at = _run_app()
        at.number_input[0].set_value(1000).run(timeout=30)  # Elongation way out of range
        at.button[0].click().run(timeout=30)
        assert not at.exception
        assert len(at.warning) > 0

    def test_zero_input_shows_error_not_crash(self):
        # st.number_input's min_value=0.0 already blocks negatives at the
        # widget level; 0 is the boundary case the app's own validation must
        # still reject (a grade can't have zero yield strength).
        at = _run_app()
        at.number_input[1].set_value(0.0).run(timeout=30)  # Yield Strength
        at.button[0].click().run(timeout=30)
        assert not at.exception
        assert len(at.error) > 0


def _switch_to_reverse(at: AppTest) -> AppTest:
    sc = at.segmented_control[0]
    reverse_option = sc.options[1]  # [FORWARD_MODE, REVERSE_MODE]
    return sc.set_value(reverse_option).run(timeout=30)


class TestReverseModeSmoke:
    def test_reverse_mode_loads_without_exception(self):
        at = _run_app()
        _switch_to_reverse(at)
        assert not at.exception

    def test_reverse_golden_path_renders_results(self):
        at = _run_app()
        _switch_to_reverse(at)
        # Key-element defaults (C/Si/Mn/P/S) are nonzero out of the box.
        at.button[0].click().run(timeout=30)
        assert not at.exception
        assert len(at.tabs) == 3
        assert len(at.dataframe) > 0

    def test_reverse_all_zero_composition_shows_error(self):
        at = _run_app()
        _switch_to_reverse(at)
        for ni in at.number_input:
            ni.set_value(0.0)
        at.run(timeout=30)
        at.button[0].click().run(timeout=30)
        assert not at.exception
        assert len(at.error) > 0

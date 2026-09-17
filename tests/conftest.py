import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data" / "raw" / "BS_EN_Dataset.xlsx"


@lru_cache(maxsize=1)
def cached_training_dataset(k: int = 75):
    """Build the expanded/normalized training dataset once per test session."""
    from src.data_pipeline import build_training_dataset

    return build_training_dataset(str(DATASET_PATH), k=k)


@lru_cache(maxsize=1)
def cached_reverse_training_dataset(k: int = 75):
    """Build the expanded/normalized reverse-direction training dataset once
    per test session."""
    from src.data_pipeline import build_reverse_training_dataset

    return build_reverse_training_dataset(str(DATASET_PATH), k=k)

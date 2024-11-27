from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROVIDED_DATA_DIR = DATA_DIR / "provided"
TRAIN_DATA_DIR = PROVIDED_DATA_DIR / "train"
TEST_DATA_DIR = PROVIDED_DATA_DIR / "test-gold"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_WITH_FEATURES_PATH = PROCESSED_DATA_DIR / "train_data_with_features.csv"
TEST_DATA_WITH_FEATURES_PATH = PROCESSED_DATA_DIR / "test_data_with_features.csv"

FEATURE_STEPS_PATH = DATA_DIR / "feature_steps.csv"
FEATURE_STEPS_WITH_IMPORTANCE_PATH = DATA_DIR / "feature_steps_with_importance.csv"

INPUT_FILENAME_PATTERN = "STS.input.{}.txt"
GS_FILENAME_PATTERN = "STS.gs.{}.txt"

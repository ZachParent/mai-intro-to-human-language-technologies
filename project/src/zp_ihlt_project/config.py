from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROVIDED_DATA_DIR = DATA_DIR / "provided"
TRAIN_DATA_DIR = PROVIDED_DATA_DIR / "train"
TEST_DATA_DIR = PROVIDED_DATA_DIR / "test-gold"

INPUT_FILENAME_PATTERN = "STS.input.{}.txt"
GS_FILENAME_PATTERN = "STS.gs.{}.txt"

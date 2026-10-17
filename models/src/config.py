from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "dataset"

MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_PATH = DATA_RAW / "UNSW_NB15_training-set.parquet"
TEST_PATH  = DATA_RAW / "UNSW_NB15_testing-set.parquet"

LEAKAGE_COLS = ["id", "srcip", "sport", "dstip", "dsport", "Stime", "Ltime"]
CAT_COLS     = ["proto", "service", "state"]

LABEL_COL      = "label"
ATTACK_CAT_COL = "attack_cat"

RANDOM_STATE      = 42
VAL_SIZE          = 0.2     # 20% train dùng làm validation
DEFAULT_THRESHOLD = 0.5
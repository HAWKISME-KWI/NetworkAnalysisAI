
import pandas as pd
from config import TRAIN_PATH, TEST_PATH, LABEL_COL, ATTACK_CAT_COL

def inspect_dataset(name, path):
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name}")
    print(f"PATH: {path}")

    if not path.exists():
        print("ERROR: File not found!")
        return None

    df = pd.read_parquet(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Kiểm tra cột bắt buộc
    required = [LABEL_COL, ATTACK_CAT_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print("Missing columns:", missing)
        return df

    print("\nLabel distribution:")
    print(df[LABEL_COL].value_counts(dropna=False))
    print("\nAttack category distribution:")
    print(df[ATTACK_CAT_COL].value_counts(dropna=False))

    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Infinite values:", df.select_dtypes(include="number")
          .isin([float("inf"), float("-inf")]).sum().sum())

    return df


train_df = inspect_dataset("TRAINING", TRAIN_PATH)
test_df = inspect_dataset("TESTING", TEST_PATH)

if train_df is not None and test_df is not None:
    print(f"\n{'=' * 60}")
    print("TRAIN / TEST SCHEMA CHECK")

    print("Same columns:", set(train_df.columns) == set(test_df.columns))
    print("Same column order:", train_df.columns.tolist() == test_df.columns.tolist())

    if set(train_df.columns) != set(test_df.columns):
        print("Only in train:", set(train_df.columns) - set(test_df.columns))
        print("Only in test:", set(test_df.columns) - set(train_df.columns))

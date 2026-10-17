
import pandas as pd
from config import TRAIN_PATH, TEST_PATH, LABEL_COL, ATTACK_CAT_COL

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Số dòng trùng hoàn toàn trong mỗi tập
print("Train duplicate rows:", train.duplicated().sum())
print("Test duplicate rows:", test.duplicated().sum())

# So sánh các dòng trên toàn bộ cột
common_cols = train.columns.intersection(test.columns).tolist()

train_rows = pd.MultiIndex.from_frame(train[common_cols])
test_rows = pd.MultiIndex.from_frame(test[common_cols])

overlap = test_rows.isin(train_rows)

print("Test rows also found in train:", overlap.sum())
print("Test overlap percentage:", round(overlap.mean() * 100, 2), "%")

# Kiểm tra các dòng trùng có nhất quán về nhãn không
feature_cols = [
    c for c in common_cols
    if c not in [LABEL_COL, ATTACK_CAT_COL]
]

train_keys = train[feature_cols].drop_duplicates()
test_keys = test[feature_cols].drop_duplicates()

print("Unique train feature rows:", len(train_keys))
print("Unique test feature rows:", len(test_keys))

# So sánh feature, bỏ các cột nhãn
feature_cols = [
    c for c in common_cols
    if c not in [LABEL_COL, ATTACK_CAT_COL]
]

train_features = pd.MultiIndex.from_frame(
    train[feature_cols].drop_duplicates()
)
test_features = pd.MultiIndex.from_frame(
    test[feature_cols]
)

feature_overlap = test_features.isin(train_features)

print(
    "Test rows with features seen in train:",
    feature_overlap.sum()
)
print(
    "Feature overlap percentage:",
    round(feature_overlap.mean() * 100, 2),
    "%"
)

# Kiểm tra các feature giống nhau nhưng label khác nhau
train_labels = train[feature_cols + [LABEL_COL]].drop_duplicates()
label_counts = train_labels.groupby(feature_cols)[LABEL_COL].nunique()
ambiguous = (label_counts > 1).sum()

print("Train feature patterns with conflicting labels:", ambiguous)

# Các feature pattern có nhiều label trong train
conflict_keys = (
    train.groupby(feature_cols)[LABEL_COL]
    .nunique()
    .loc[lambda s: s > 1]
    .index
)

conflict_df = train.set_index(feature_cols).loc[
    conflict_keys
].reset_index()

print("\nConflicting feature patterns:", len(conflict_keys))
print("\nExamples of conflicting labels:")

print(
    conflict_df[
        feature_cols + [LABEL_COL, ATTACK_CAT_COL]
    ]
    .sort_values(feature_cols)
    .head(20)
    .to_string(index=False)
)

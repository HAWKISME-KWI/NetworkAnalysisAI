import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from xgboost import XGBClassifier

from config import (
    TRAIN_PATH, TEST_PATH, MODELS_DIR, ATTACK_CAT_COL,
    RANDOM_STATE, VAL_SIZE,
)
from preprocess import (
    load_raw, drop_leakage, fit_preprocessors, transform, save_artifacts,
)


COARSE_MAP = {
    'Normal':         'Normal',
    'Generic':        'Generic',
    'Exploits':       'Exploitation',
    'Fuzzers':        'Exploitation',
    'Reconnaissance': 'Reconnaissance',
    'Analysis':       'Reconnaissance',
    'DoS':            'DoS',
    'Backdoor':       'Backdoor',
    'Shellcode':      'Backdoor',
    'Worms':          'Backdoor',
}

MAX_WEIGHT = 10.0


def compute_capped_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    raw = {int(c): len(y) / (len(classes) * int(cnt))
           for c, cnt in zip(classes, counts)}
    capped = {c: min(w, MAX_WEIGHT) for c, w in raw.items()}
    return np.array([capped[int(c)] for c in y]), capped, raw


def build_model(n_classes):
    return XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        objective='multi:softprob', num_class=n_classes,
        tree_method='hist', n_jobs=-1, random_state=RANDOM_STATE,
        eval_metric='mlogloss',
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Loading data...")
    train_full = drop_leakage(load_raw(TRAIN_PATH))
    test_df    = drop_leakage(load_raw(TEST_PATH))
    train_full[ATTACK_CAT_COL] = train_full[ATTACK_CAT_COL].astype(str).str.strip()
    test_df[ATTACK_CAT_COL]    = test_df[ATTACK_CAT_COL].astype(str).str.strip()

    train_full['_coarse'] = train_full[ATTACK_CAT_COL].map(COARSE_MAP)
    test_df['_coarse']    = test_df[ATTACK_CAT_COL].map(COARSE_MAP)
    assert train_full['_coarse'].notna().all()
    assert test_df['_coarse'].notna().all()

    print(f"[*] Train: {train_full.shape} | Test: {test_df.shape}")

    le = LabelEncoder()
    y_all  = le.fit_transform(train_full['_coarse'])
    y_test = le.transform(test_df['_coarse'])
    n_classes = len(le.classes_)

    print(f"\n[*] Coarse classes ({n_classes}): {list(le.classes_)}")
    print("[*] Train distribution:")
    for i, c in enumerate(le.classes_):
        print(f"   {c:15s}: {(y_all == i).sum():>7d}")
    print("[*] Test distribution:")
    for i, c in enumerate(le.classes_):
        print(f"   {c:15s}: {(y_test == i).sum():>7d}")

    # Split
    train_sub, val_df = train_test_split(
        train_full, test_size=VAL_SIZE,
        stratify=y_all, random_state=RANDOM_STATE,
    )
    y_tr  = le.transform(train_sub['_coarse'])
    y_val = le.transform(val_df['_coarse'])
    print(f"\n[*] Split -> train_sub {train_sub.shape} | val {val_df.shape}")

    # Fit preprocessor
    artifacts, X_tr, _ = fit_preprocessors(train_sub)
    X_val, _, _ = transform(val_df, artifacts)

    # Weights capped
    w_tr, w_map, w_raw = compute_capped_weights(y_tr)
    print("\n[*] Class weights (raw -> capped):")
    for i, c in enumerate(le.classes_):
        print(f"   {c:15s}: {w_raw[i]:7.2f} -> {w_map[i]:6.2f}")

    # Train on val split
    print("\n[*] Training XGBoost (coarse)...")
    model = build_model(n_classes)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    y_val_pred = model.predict(X_val)
    val_macro_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\n===== VAL (coarse) =====")
    print(f"Accuracy: {val_acc:.4f} | Macro-F1: {val_macro_f1:.4f}")
    print(classification_report(
        y_val, y_val_pred, labels=range(n_classes),
        target_names=list(le.classes_), digits=4, zero_division=0,
    ))

    # Refit full
    print("\n[*] Refitting on full train...")
    artifacts_full, X_train_full, _ = fit_preprocessors(train_full)
    save_artifacts(artifacts_full, MODELS_DIR)
    w_full, _, _ = compute_capped_weights(y_all)

    model_full = build_model(n_classes)
    model_full.fit(X_train_full, y_all, sample_weight=w_full)

    # Test eval
    print("\n[*] Evaluating on test...")
    X_test, _, _ = transform(test_df, artifacts_full)
    y_test_pred = model_full.predict(X_test)

    test_macro_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n===== TEST (coarse) =====")
    print(f"Accuracy: {test_acc:.4f} | Macro-F1: {test_macro_f1:.4f}")
    print(classification_report(
        y_test, y_test_pred, labels=range(n_classes),
        target_names=list(le.classes_), digits=4, zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_test_pred, labels=range(n_classes))
    print("\nConfusion matrix [rows=actual, cols=pred]:")
    print("classes:", list(le.classes_))
    print(cm)

    # Save
    joblib.dump(model_full, MODELS_DIR / "model_coarse.pkl")
    joblib.dump(le,         MODELS_DIR / "label_encoder_coarse.pkl")

    metadata = {
        "task": "coarse_multiclass",
        "model_name": "XGBoost",
        "classes": list(le.classes_),
        "n_classes": n_classes,
        "mapping": COARSE_MAP,
        "val_macro_f1":  float(val_macro_f1),
        "val_accuracy":  float(val_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_accuracy": float(test_acc),
        "class_weights": {c: float(w_map[i]) for i, c in enumerate(le.classes_)},
    }
    with open(MODELS_DIR / "metadata_coarse.json", "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(metadata, MODELS_DIR / "metadata_coarse.pkl")

    print("\n[✓] Done.")
    print(f"   model    -> {MODELS_DIR / 'model_coarse.pkl'}")
    print(f"   encoder  -> {MODELS_DIR / 'label_encoder_coarse.pkl'}")
    print(f"   metadata -> {MODELS_DIR / 'metadata_coarse.json'}")


if __name__ == "__main__":
    main()
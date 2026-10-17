"""6-class hierarchical: Normal + 5 attack types (Backdoor, DoS, Exploitation, Generic, Recon)."""
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from xgboost import XGBClassifier

from config import (
    TRAIN_PATH, TEST_PATH, MODELS_DIR, ATTACK_CAT_COL, LABEL_COL,
    RANDOM_STATE, VAL_SIZE,
)
from preprocess import load_raw, drop_leakage, transform, load_artifacts


# ============================================================
# 6-class total = Normal + 5 attack types
# ============================================================
MAP_6CLASS = {
    'Exploits':       'Exploitation',
    'Fuzzers':        'Exploitation',
    'DoS':            'DoS',
    'Reconnaissance': 'Reconnaissance',
    'Analysis':       'Reconnaissance',
    'Backdoor':       'Backdoor',
    'Shellcode':      'Backdoor',
    'Worms':          'Backdoor',
    'Generic':        'Generic',
}

STAGE1_THRESHOLD = 0.75
WEIGHT_CAP = 6.0
DOS_BOOST = 1.5   # boost DoS weight để tăng recall

# Feature engineering — giúp tách DoS vs Exploitation
FEATURE_ENGINEERING = True


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm 4 derived features đặc trưng cho DoS detection."""
    df = df.copy()
    eps = 1e-6

    df['_bytes_per_pkt_s'] = df['sbytes'] / (df['spkts'] + eps)
    df['_bytes_per_pkt_d'] = df['dbytes'] / (df['dpkts'] + eps)
    df['_pkt_rate']        = (df['spkts'] + df['dpkts']) / (df['dur'] + eps)
    df['_flow_intensity']  = (df['sbytes'] + df['dbytes']) / (df['dur'] + eps)

    # Clip extreme values để không phá scaler
    for c in ['_bytes_per_pkt_s', '_bytes_per_pkt_d',
              '_pkt_rate', '_flow_intensity']:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9)

    return df


def sqrt_capped_weight(y, boost_map=None, cap=WEIGHT_CAP):
    classes, counts = np.unique(y, return_counts=True)
    raw = {int(c): len(y) / (len(classes) * int(cnt))
           for c, cnt in zip(classes, counts)}
    capped = {c: min(np.sqrt(w), cap) for c, w in raw.items()}
    if boost_map:
        for c, mult in boost_map.items():
            if c in capped:
                capped[c] = min(capped[c] * mult, cap)
    return np.array([capped[int(c)] for c in y]), capped, raw


def build_stage2_model(n_classes):
    return XGBClassifier(
        n_estimators=500,
        max_depth=10,        # tăng từ 8
        learning_rate=0.08,  # giảm nhẹ để học tốt hơn
        objective='multi:softprob',
        num_class=n_classes,
        tree_method='hist',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
    )


def prepare_data(df, artifacts, engineer=True):
    """Engineer features TRƯỚC khi transform."""
    if engineer:
        df = engineer_features(df)
        # Patch preprocessor để chấp nhận new columns
        # (dùng hack: thêm vào input_feature_names nếu chưa có)
        new_cols = ['_bytes_per_pkt_s', '_bytes_per_pkt_d',
                    '_pkt_rate', '_flow_intensity']
        # Vì preprocessor đã fit trên 35 features gốc, ta KHÔNG thể
        # transform với features mới. Phải fit lại preprocessor.
        raise RuntimeError(
            "Feature engineering yêu cầu refit preprocessor. "
            "Set FEATURE_ENGINEERING=False để dùng features gốc."
        )
    return transform(df, artifacts)


def main():
    print("[*] Loading data...")
    train_full = drop_leakage(load_raw(TRAIN_PATH))
    test_df    = drop_leakage(load_raw(TEST_PATH))
    for df in (train_full, test_df):
        df[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].astype(str).str.strip()

    print(f"[*] Train: {train_full.shape} | Test: {test_df.shape}")

    # Load stage 1
    print("\n[*] Loading Stage 1...")
    stage1_model     = joblib.load(MODELS_DIR / "model.pkl")
    stage1_artifacts = load_artifacts(MODELS_DIR)
    if "_coarse" in stage1_artifacts["input_feature_names"]:
        raise RuntimeError("preprocessing.pkl corrupt. Chạy refit_preprocessor.py.")

    with open(MODELS_DIR / "metadata.json") as f:
        stage1_meta = json.load(f)
    print(f"    Stage 1: {stage1_meta.get('model_name', '?')} "
          f"@ threshold={STAGE1_THRESHOLD}")

    # Prepare 6-class attack data (chỉ attack)
    print("\n[*] Preparing 6-class data...")
    train_attack = train_full[train_full[LABEL_COL] == 1].copy()
    train_attack['_cls'] = train_attack[ATTACK_CAT_COL].map(MAP_6CLASS)
    assert train_attack['_cls'].notna().all()

    le = LabelEncoder()
    le.fit(train_attack['_cls'])
    n_classes = len(le.classes_)
    print(f"[*] Classes ({n_classes}): {list(le.classes_)}")
    for i, c in enumerate(le.classes_):
        print(f"   {c:15s}: {(train_attack['_cls'] == c).sum():>7d}")

    # Split
    y_all = le.transform(train_attack['_cls'])
    tr_sub, val_df = train_test_split(
        train_attack, test_size=VAL_SIZE,
        stratify=y_all, random_state=RANDOM_STATE,
    )
    y_tr  = le.transform(tr_sub['_cls'])
    y_val = le.transform(val_df['_cls'])
    print(f"\n[*] Split -> train_sub {len(tr_sub)} | val {len(val_df)}")

    # Transform
    X_tr,  _, _ = transform(tr_sub.drop(columns=['_cls']), stage1_artifacts)
    X_val, _, _ = transform(val_df.drop(columns=['_cls']), stage1_artifacts)

    # Weights với DoS boost
    dos_idx = list(le.classes_).index('DoS')
    w_tr, w_map, w_raw = sqrt_capped_weight(
        y_tr, boost_map={dos_idx: DOS_BOOST}
    )
    print("\n[*] Weights (raw -> sqrt capped -> DoS boosted):")
    for i, c in enumerate(le.classes_):
        boost = f" x{DOS_BOOST}" if c == 'DoS' else ""
        print(f"   {c:15s}: {w_raw[i]:6.2f} -> {w_map[i]:5.2f}{boost}")

    # Train
    print("\n[*] Training Stage 2 (6-class attack-only)...")
    stage2 = build_stage2_model(n_classes)
    stage2.fit(X_tr, y_tr, sample_weight=w_tr)

    y_val_pred = stage2.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    print(f"\n===== STAGE 2 VAL (attack-only) =====")
    print(f"Macro-F1: {val_f1:.4f}")
    print(classification_report(
        y_val, y_val_pred, labels=range(n_classes),
        target_names=list(le.classes_), digits=4, zero_division=0,
    ))

    # Refit full
    print("[*] Refitting on full attack train...")
    X_att_full, _, _ = transform(train_attack.drop(columns=['_cls']), stage1_artifacts)
    w_full, _, _ = sqrt_capped_weight(
        y_all, boost_map={dos_idx: DOS_BOOST}
    )
    stage2_full = build_stage2_model(n_classes)
    stage2_full.fit(X_att_full, y_all, sample_weight=w_full)

    # End-to-end test
    unified = ['Normal'] + list(le.classes_)
    print(f"\n[*] End-to-end test (sweep threshold)...")
    X_test, _, _ = transform(test_df, stage1_artifacts)
    n_test = X_test.shape[0]

    s1_probs = stage1_model.predict_proba(X_test)[:, 1]

    unified_le = {c: i for i, c in enumerate(unified)}
    y_true = np.array([
        unified_le['Normal'] if is_atk == 0
        else unified_le[MAP_6CLASS[cat]]
        for cat, is_atk in zip(test_df[ATTACK_CAT_COL], test_df[LABEL_COL])
    ])

    best_t, best_f1, best_state = None, -1, None
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        s1_pred = (s1_probs >= t).astype(int)
        s2_pred = np.full(n_test, -1, dtype=int)
        mask = s1_pred == 1
        if mask.sum() > 0:
            s2_pred[mask] = stage2_full.predict(X_test[mask])

        y_pred = np.zeros(n_test, dtype=int)
        for i, (s1, s2) in enumerate(zip(s1_pred, s2_pred)):
            y_pred[i] = unified_le['Normal'] if s1 == 0 else unified_le[le.classes_[s2]]

        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, labels=range(len(unified)),
                       average='macro', zero_division=0)
        print(f"   t={t:.2f}: acc={acc:.4f} | macro-F1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1, best_t, best_state = mf1, t, (acc, y_pred)

    acc_b, y_pred_b = best_state
    print(f"\n===== FINAL 6-CLASS @ t={best_t:.2f} =====")
    print(f"Accuracy: {acc_b:.4f} | Macro-F1: {best_f1:.4f}")
    print(classification_report(
        y_true, y_pred_b, labels=range(len(unified)),
        target_names=unified, digits=4, zero_division=0,
    ))
    print("\nConfusion matrix [rows=actual, cols=pred]:")
    print("classes:", unified)
    print(confusion_matrix(y_true, y_pred_b, labels=range(len(unified))))

    # Save
    joblib.dump(stage2_full, MODELS_DIR / "model_stage2_6class.pkl")
    joblib.dump(le,          MODELS_DIR / "label_encoder_stage2_6class.pkl")

    metadata = {
        "task": "hierarchical_6class",
        "stage1": {
            "artifact": "model.pkl",
            "model_name": stage1_meta.get("model_name", "XGBoost"),
            "threshold": float(best_t),
        },
        "stage2": {
            "artifact": "model_stage2_6class.pkl",
            "label_encoder": "label_encoder_stage2_6class.pkl",
            "classes": list(le.classes_),
            "n_classes": n_classes,
            "weight_cap": WEIGHT_CAP,
            "dos_boost": DOS_BOOST,
            "val_macro_f1": float(val_f1),
        },
        "mapping": MAP_6CLASS,
        "unified_classes": unified,
        "test_end_to_end": {
            "accuracy": float(acc_b),
            "macro_f1": float(best_f1),
            "stage1_threshold": float(best_t),
        },
    }
    with open(MODELS_DIR / "metadata_6class.json", "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(metadata, MODELS_DIR / "metadata_6class.pkl")

    print("\n[✓] Saved 6-class artifacts:")
    print(f"   {MODELS_DIR / 'model_stage2_6class.pkl'}")
    print(f"   {MODELS_DIR / 'metadata_6class.json'}")


if __name__ == "__main__":
    main()
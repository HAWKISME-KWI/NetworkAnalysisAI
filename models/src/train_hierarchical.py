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
from preprocess import (
    load_raw, drop_leakage, transform, load_artifacts,
)


COARSE_ATTACK_MAP = {
    'Exploits':       'Exploitation',
    'Fuzzers':        'Exploitation',
    'Reconnaissance': 'Reconnaissance',
    'Analysis':       'Reconnaissance',
    'DoS':            'DoS',
    'Backdoor':       'Backdoor',
    'Shellcode':      'Backdoor',
    'Worms':          'Backdoor',
    'Generic':        'Generic',
}

STAGE1_THRESHOLD = 0.5
WEIGHT_CAP = 5.0


def sqrt_capped_weight(y, cap=WEIGHT_CAP):
    classes, counts = np.unique(y, return_counts=True)
    raw = {int(c): len(y) / (len(classes) * int(cnt))
           for c, cnt in zip(classes, counts)}
    capped = {c: min(np.sqrt(w), cap) for c, w in raw.items()}
    return np.array([capped[int(c)] for c in y]), capped, raw


def build_stage2_model(n_classes):
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
    for df in (train_full, test_df):
        df[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].astype(str).str.strip()

    print(f"[*] Train: {train_full.shape} | Test: {test_df.shape}")

    # ============================================================
    # Load Stage 1
    # ============================================================
    print("\n[*] Loading Stage 1 artifacts...")
    stage1_model     = joblib.load(MODELS_DIR / "model.pkl")
    stage1_artifacts = load_artifacts(MODELS_DIR)

    # GUARD: chặn preprocessor corrupt (từng bị multi_train_coarse ghi đè)
    if "_coarse" in stage1_artifacts["input_feature_names"]:
        raise RuntimeError(
            "preprocessing.pkl is corrupted (expects '_coarse' as feature). "
            "Chạy `py refit_preprocessor.py` trước."
        )

    with open(MODELS_DIR / "metadata.json") as f:
        stage1_meta = json.load(f)
    print(f"    Stage 1: {stage1_meta.get('model_name', '?')} @ "
          f"threshold={STAGE1_THRESHOLD}")

    # ============================================================
    # Stage 2 data: chỉ attack
    # ============================================================
    print("\n[*] Preparing Stage 2 data (attack-only)...")
    train_attack = train_full[train_full[LABEL_COL] == 1].copy()
    train_attack['_coarse'] = train_attack[ATTACK_CAT_COL].map(COARSE_ATTACK_MAP)
    assert train_attack['_coarse'].notna().all(), "Unknown attack category!"

    print(f"    Attack train: {len(train_attack)}")

    le_stage2 = LabelEncoder()
    le_stage2.fit(train_attack['_coarse'])
    n_classes = len(le_stage2.classes_)
    print(f"\n[*] Stage 2 classes ({n_classes}): {list(le_stage2.classes_)}")
    for i, c in enumerate(le_stage2.classes_):
        n = (train_attack['_coarse'] == c).sum()
        print(f"   {c:15s}: {n:>7d}")

    # ============================================================
    # Split attack-only -> train_sub / val
    # ============================================================
    y_attack_all = le_stage2.transform(train_attack['_coarse'])
    tr_sub, val_df = train_test_split(
        train_attack, test_size=VAL_SIZE,
        stratify=y_attack_all, random_state=RANDOM_STATE,
    )
    y_tr  = le_stage2.transform(tr_sub['_coarse'])
    y_val = le_stage2.transform(val_df['_coarse'])
    print(f"\n[*] Attack split -> train_sub {len(tr_sub)} | val {len(val_df)}")

    # ============================================================
    # Transform — drop _coarse để pass validation
    # ============================================================
    print("\n[*] Transforming với Stage 1 preprocessor (shared)...")
    X_tr,  _, _ = transform(tr_sub.drop(columns=['_coarse']), stage1_artifacts)
    X_val, _, _ = transform(val_df.drop(columns=['_coarse']), stage1_artifacts)

    # ============================================================
    # Train Stage 2 on train_sub
    # ============================================================
    w_tr, w_map, w_raw = sqrt_capped_weight(y_tr)
    print("\n[*] Stage 2 weights (raw -> sqrt capped):")
    for i, c in enumerate(le_stage2.classes_):
        print(f"   {c:15s}: {w_raw[i]:7.2f} -> {w_map[i]:6.2f}")

    print("\n[*] Training Stage 2 XGBoost...")
    stage2_model = build_stage2_model(n_classes)
    stage2_model.fit(X_tr, y_tr, sample_weight=w_tr)

    y_val_pred = stage2_model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    print(f"\n===== STAGE 2 VAL (attack-only) =====")
    print(f"Macro-F1: {val_f1:.4f}")
    print(classification_report(
        y_val, y_val_pred, labels=range(n_classes),
        target_names=list(le_stage2.classes_), digits=4, zero_division=0,
    ))

    # ============================================================
    # Refit Stage 2 trên toàn bộ attack train
    # ============================================================
    print("\n[*] Refitting Stage 2 on full attack train...")
    X_attack_full, _, _ = transform(
        train_attack.drop(columns=['_coarse']), stage1_artifacts
    )
    w_full, _, _ = sqrt_capped_weight(y_attack_all)

    stage2_final = build_stage2_model(n_classes)
    stage2_final.fit(X_attack_full, y_attack_all, sample_weight=w_full)

    # ============================================================
    # Evaluate END-TO-END trên test @ threshold 0.5
    # ============================================================
    print(f"\n[*] Evaluating end-to-end (stage1_threshold={STAGE1_THRESHOLD})...")
    X_test, _, _ = transform(test_df, stage1_artifacts)

    n_test = X_test.shape[0]  # sparse-safe

    stage1_probs = stage1_model.predict_proba(X_test)[:, 1]
    stage1_pred  = (stage1_probs >= STAGE1_THRESHOLD).astype(int)

    attack_mask = stage1_pred == 1
    n_attack_pred = int(attack_mask.sum())
    print(f"    Stage 1 predicts attack: {n_attack_pred}/{n_test} "
          f"({100 * n_attack_pred / n_test:.1f}%)")

    stage2_pred = np.full(n_test, -1, dtype=int)
    if n_attack_pred > 0:
        stage2_pred[attack_mask] = stage2_final.predict(X_test[attack_mask])

    unified_classes = ['Normal'] + list(le_stage2.classes_)
    unified_le = {c: i for i, c in enumerate(unified_classes)}

    y_true_unified = np.array([
        unified_le['Normal'] if is_atk == 0
        else unified_le[COARSE_ATTACK_MAP[cat]]
        for cat, is_atk in zip(test_df[ATTACK_CAT_COL], test_df[LABEL_COL])
    ])

    y_pred_unified = np.zeros(n_test, dtype=int)
    for i, (s1, s2) in enumerate(zip(stage1_pred, stage2_pred)):
        if s1 == 0:
            y_pred_unified[i] = unified_le['Normal']
        else:
            y_pred_unified[i] = unified_le[le_stage2.classes_[s2]]

    acc  = accuracy_score(y_true_unified, y_pred_unified)
    mf1  = f1_score(y_true_unified, y_pred_unified,
                    labels=range(len(unified_classes)),
                    average='macro', zero_division=0)

    print(f"\n===== END-TO-END TEST @ {STAGE1_THRESHOLD} =====")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
    print(classification_report(
        y_true_unified, y_pred_unified,
        labels=range(len(unified_classes)),
        target_names=unified_classes, digits=4, zero_division=0,
    ))
    print("\nConfusion matrix [rows=actual, cols=pred]:")
    print("classes:", unified_classes)
    print(confusion_matrix(y_true_unified, y_pred_unified,
                           labels=range(len(unified_classes))))

    # ============================================================
    # Save
    # ============================================================
    joblib.dump(stage2_final, MODELS_DIR / "model_stage2.pkl")
    joblib.dump(le_stage2,    MODELS_DIR / "label_encoder_stage2.pkl")

    metadata = {
        "task": "hierarchical_2stage",
        "stage1": {
            "model_name": stage1_meta.get("model_name", "XGBoost"),
            "artifact":   "model.pkl",
            "threshold":  STAGE1_THRESHOLD,
        },
        "stage2": {
            "model_name": "XGBoost",
            "artifact":   "model_stage2.pkl",
            "classes":    list(le_stage2.classes_),
            "n_classes":  n_classes,
            "weight_cap": WEIGHT_CAP,
            "val_macro_f1_attack_only": float(val_f1),
        },
        "coarse_mapping":  COARSE_ATTACK_MAP,
        "unified_classes": unified_classes,
        "test_end_to_end": {
            "stage1_threshold": STAGE1_THRESHOLD,
            "accuracy": float(acc),
            "macro_f1": float(mf1),
        },
    }
    with open(MODELS_DIR / "metadata_hierarchical.json", "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(metadata, MODELS_DIR / "metadata_hierarchical.pkl")

    print("\n[✓] Done.")
    print(f"   stage2 model   -> {MODELS_DIR / 'model_stage2.pkl'}")
    print(f"   stage2 encoder -> {MODELS_DIR / 'label_encoder_stage2.pkl'}")
    print(f"   metadata       -> {MODELS_DIR / 'metadata_hierarchical.json'}")


if __name__ == "__main__":
    main()
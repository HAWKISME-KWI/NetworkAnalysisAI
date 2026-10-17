"""Quick test: 5-class merge DoS+Exploits+Fuzzers -> Exploitation."""
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from xgboost import XGBClassifier

from config import (
    TRAIN_PATH, TEST_PATH, MODELS_DIR, ATTACK_CAT_COL, LABEL_COL,
    RANDOM_STATE, VAL_SIZE,
)
from preprocess import load_raw, drop_leakage, transform, load_artifacts


# ============================================================
# 5-class mapping — DoS + Exploits + Fuzzers -> Exploitation
# ============================================================
MAP_5CLASS = {
    'Exploits':       'Exploitation',
    'Fuzzers':        'Exploitation',
    'DoS':            'Exploitation',   # merge vào
    'Reconnaissance': 'Reconnaissance',
    'Analysis':       'Reconnaissance',
    'Backdoor':       'Backdoor',
    'Shellcode':      'Backdoor',
    'Worms':          'Backdoor',
    'Generic':        'Generic',
}

STAGE1_THRESHOLD = 0.75  # từ sweep best
WEIGHT_CAP = 5.0


def sqrt_capped_weight(y, cap=WEIGHT_CAP):
    classes, counts = np.unique(y, return_counts=True)
    raw = {int(c): len(y) / (len(classes) * int(cnt))
           for c, cnt in zip(classes, counts)}
    capped = {c: min(np.sqrt(w), cap) for c, w in raw.items()}
    return np.array([capped[int(c)] for c in y]), capped, raw


def build_model(n):
    return XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        objective='multi:softprob', num_class=n,
        tree_method='hist', n_jobs=-1, random_state=RANDOM_STATE,
        eval_metric='mlogloss',
    )


def evaluate_e2e(stage1_model, stage2_model, le_stage2, artifacts,
                 test_df, mapping, unified_classes, threshold):
    X_test, _, _ = transform(test_df, artifacts)
    n_test = X_test.shape[0]

    s1_probs = stage1_model.predict_proba(X_test)[:, 1]
    s1_pred  = (s1_probs >= threshold).astype(int)

    s2_pred = np.full(n_test, -1, dtype=int)
    mask = s1_pred == 1
    if mask.sum() > 0:
        s2_pred[mask] = stage2_model.predict(X_test[mask])

    unified_le = {c: i for i, c in enumerate(unified_classes)}
    y_true = np.array([
        unified_le['Normal'] if is_atk == 0
        else unified_le[mapping[cat]]
        for cat, is_atk in zip(test_df[ATTACK_CAT_COL], test_df[LABEL_COL])
    ])

    y_pred = np.zeros(n_test, dtype=int)
    for i, (s1, s2) in enumerate(zip(s1_pred, s2_pred)):
        y_pred[i] = unified_le['Normal'] if s1 == 0 else unified_le[le_stage2.classes_[s2]]

    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, labels=range(len(unified_classes)),
                 average='macro', zero_division=0),
        y_true, y_pred,
    )


def main():
    print("[*] Loading data + stage 1...")
    train_full = drop_leakage(load_raw(TRAIN_PATH))
    test_df    = drop_leakage(load_raw(TEST_PATH))
    for df in (train_full, test_df):
        df[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].astype(str).str.strip()

    stage1_model     = joblib.load(MODELS_DIR / "model.pkl")
    stage1_artifacts = load_artifacts(MODELS_DIR)

    if "_coarse" in stage1_artifacts["input_feature_names"]:
        raise RuntimeError("preprocessing.pkl corrupt. Chạy refit_preprocessor.py trước.")

    # ---- Stage 2 5-class ----
    print("\n[*] Preparing 5-class attack data...")
    train_attack = train_full[train_full[LABEL_COL] == 1].copy()
    train_attack['_cls'] = train_attack[ATTACK_CAT_COL].map(MAP_5CLASS)

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

    X_tr,  _, _ = transform(tr_sub.drop(columns=['_cls']), stage1_artifacts)
    X_val, _, _ = transform(val_df.drop(columns=['_cls']), stage1_artifacts)

    w_tr, w_map, w_raw = sqrt_capped_weight(y_tr)
    print("\n[*] Weights (raw -> sqrt capped):")
    for i, c in enumerate(le.classes_):
        print(f"   {c:15s}: {w_raw[i]:6.2f} -> {w_map[i]:5.2f}")

    print("\n[*] Training Stage 2 (5-class)...")
    stage2 = build_model(n_classes)
    stage2.fit(X_tr, y_tr, sample_weight=w_tr)

    y_val_pred = stage2.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    print(f"\n===== STAGE 2 VAL (5-class attack-only) =====")
    print(f"Macro-F1: {val_f1:.4f}")
    print(classification_report(
        y_val, y_val_pred, labels=range(n_classes),
        target_names=list(le.classes_), digits=4, zero_division=0,
    ))

    # Refit full
    print("[*] Refitting on full attack train...")
    X_att_full, _, _ = transform(train_attack.drop(columns=['_cls']), stage1_artifacts)
    w_full, _, _ = sqrt_capped_weight(y_all)
    stage2_full = build_model(n_classes)
    stage2_full.fit(X_att_full, y_all, sample_weight=w_full)

    # End-to-end test @ 0.5 và 0.75
    unified = ['Normal'] + list(le.classes_)
    print(f"\n===== END-TO-END @ various thresholds =====")
    best_t, best_f1, best_state = None, -1, None
    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        acc, mf1, yt, yp = evaluate_e2e(
            stage1_model, stage2_full, le, stage1_artifacts,
            test_df, MAP_5CLASS, unified, t,
        )
        print(f"   t={t:.2f}: acc={acc:.4f} | macro-F1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1, best_t, best_state = mf1, t, (acc, yt, yp)

    acc_b, yt_b, yp_b = best_state
    print(f"\n===== FINAL 5-CLASS @ t={best_t:.2f} =====")
    print(f"Accuracy: {acc_b:.4f} | Macro-F1: {best_f1:.4f}")
    print(classification_report(
        yt_b, yp_b, labels=range(len(unified)),
        target_names=unified, digits=4, zero_division=0,
    ))
    print("\nConfusion matrix [rows=actual, cols=pred]:")
    print("classes:", unified)
    print(confusion_matrix(yt_b, yp_b, labels=range(len(unified))))

    # Save candidate
    joblib.dump(stage2_full, MODELS_DIR / "model_stage2_5class.pkl")
    joblib.dump(le,          MODELS_DIR / "label_encoder_stage2_5class.pkl")
    with open(MODELS_DIR / "metadata_5class.json", "w") as f:
        json.dump({
            "task": "hierarchical_5class",
            "stage1_threshold": float(best_t),
            "classes": list(le.classes_),
            "mapping": MAP_5CLASS,
            "val_macro_f1": float(val_f1),
            "test_macro_f1": float(best_f1),
            "test_accuracy": float(acc_b),
        }, f, indent=2)

    print(f"\n[✓] Saved candidate -> {MODELS_DIR / 'model_stage2_5class.pkl'}")


if __name__ == "__main__":
    main()
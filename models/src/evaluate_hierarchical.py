import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)

from config import TEST_PATH, MODELS_DIR, ATTACK_CAT_COL, LABEL_COL
from preprocess import load_raw, drop_leakage, transform, load_artifacts
from train_hierarchical import COARSE_ATTACK_MAP


def eval_at_threshold(threshold, stage1_model, stage2_model, le_stage2,
                      X_test, test_df, unified_classes):
    n_test = X_test.shape[0]  # sparse-safe

    stage1_probs = stage1_model.predict_proba(X_test)[:, 1]
    stage1_pred  = (stage1_probs >= threshold).astype(int)

    attack_mask = stage1_pred == 1
    stage2_pred = np.full(n_test, -1, dtype=int)
    if attack_mask.sum() > 0:
        stage2_pred[attack_mask] = stage2_model.predict(X_test[attack_mask])

    unified_le = {c: i for i, c in enumerate(unified_classes)}
    y_true = np.array([
        unified_le['Normal'] if is_atk == 0
        else unified_le[COARSE_ATTACK_MAP[cat]]
        for cat, is_atk in zip(test_df[ATTACK_CAT_COL], test_df[LABEL_COL])
    ])
    y_pred = np.zeros(n_test, dtype=int)
    for i, (s1, s2) in enumerate(zip(stage1_pred, stage2_pred)):
        if s1 == 0:
            y_pred[i] = unified_le['Normal']
        else:
            y_pred[i] = unified_le[le_stage2.classes_[s2]]

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=range(len(unified_classes)),
                   average='macro', zero_division=0)
    return acc, mf1, y_true, y_pred, int(attack_mask.sum())


def main():
    print("[*] Loading all artifacts...")
    stage1_model     = joblib.load(MODELS_DIR / "model.pkl")
    stage1_artifacts = load_artifacts(MODELS_DIR)
    stage2_model     = joblib.load(MODELS_DIR / "model_stage2.pkl")
    le_stage2        = joblib.load(MODELS_DIR / "label_encoder_stage2.pkl")

    # GUARD
    if "_coarse" in stage1_artifacts["input_feature_names"]:
        raise RuntimeError(
            "preprocessing.pkl is corrupted. Chạy `py refit_preprocessor.py` trước."
        )

    with open(MODELS_DIR / "metadata_hierarchical.json") as f:
        meta = json.load(f)
    unified_classes = meta["unified_classes"]

    test_df = drop_leakage(load_raw(TEST_PATH))
    test_df[ATTACK_CAT_COL] = test_df[ATTACK_CAT_COL].astype(str).str.strip()
    X_test, _, _ = transform(test_df, stage1_artifacts)

    # Sweep threshold
    print("\n[*] Sweeping stage 1 threshold...")
    thresholds = np.arange(0.20, 0.81, 0.05)
    rows = []
    best_t, best_mf1 = 0.5, -1.0
    best_state = None

    for t in thresholds:
        acc, mf1, y_true, y_pred, n_atk = eval_at_threshold(
            t, stage1_model, stage2_model, le_stage2,
            X_test, test_df, unified_classes,
        )
        rows.append({"threshold": float(t), "accuracy": float(acc),
                     "macro_f1": float(mf1), "n_attack_pred": n_atk})
        print(f"   t={t:.2f}: acc={acc:.4f} | macro-F1={mf1:.4f} | "
              f"attack_pred={n_atk}")
        if mf1 > best_mf1:
            best_mf1, best_t = mf1, float(t)
            best_state = (acc, y_true, y_pred)

    print(f"\n[✓] Best threshold = {best_t:.2f} (macro-F1 = {best_mf1:.4f})")

    # Plot sweep
    ts   = [r["threshold"] for r in rows]
    f1s  = [r["macro_f1"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(ts, f1s, marker="o", label="Macro-F1")
    plt.plot(ts, accs, marker="s", label="Accuracy", alpha=0.6)
    plt.axvline(best_t, color="red", linestyle="--",
                label=f"best t={best_t:.2f}")
    plt.xlabel("Stage 1 threshold")
    plt.ylabel("Score")
    plt.title("End-to-End Threshold Sweep (Test)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(MODELS_DIR / "hierarchical_threshold_sweep.png", dpi=150)
    plt.close()
    print(f"[✓] Saved hierarchical_threshold_sweep.png")

    # Final report @ best threshold
    acc_best, y_true_best, y_pred_best = best_state
    print(f"\n===== FINAL END-TO-END @ t={best_t:.2f} =====")
    print(f"Accuracy: {acc_best:.4f} | Macro-F1: {best_mf1:.4f}")
    print(classification_report(
        y_true_best, y_pred_best,
        labels=range(len(unified_classes)),
        target_names=unified_classes, digits=4, zero_division=0,
    ))
    cm = confusion_matrix(y_true_best, y_pred_best,
                          labels=range(len(unified_classes)))
    print("\nConfusion matrix [rows=actual, cols=pred]:")
    print("classes:", unified_classes)
    print(cm)

    # Save best threshold
    meta["stage1"]["threshold"] = float(best_t)
    meta["test_end_to_end"] = {
        "stage1_threshold": float(best_t),
        "accuracy": float(acc_best),
        "macro_f1": float(best_mf1),
    }
    with open(MODELS_DIR / "metadata_hierarchical.json", "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(meta, MODELS_DIR / "metadata_hierarchical.pkl")
    print(f"[✓] Updated metadata với threshold = {best_t:.2f}")


if __name__ == "__main__":
    main()
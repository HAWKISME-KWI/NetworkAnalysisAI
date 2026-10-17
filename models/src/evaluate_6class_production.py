"""Production eval: selective prediction (proper metric)."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)

from config import TEST_PATH, MODELS_DIR, ATTACK_CAT_COL, LABEL_COL
from preprocess import load_raw, drop_leakage, transform, load_artifacts
from test_6class import MAP_6CLASS


def evaluate_selective(stage1_model, stage2_model, le_stage2,
                       X_test, y_true, unified_classes,
                       stage1_threshold, conf_threshold):
    """Selective prediction: F1 chỉ trên mẫu đã phân loại."""
    n_test = X_test.shape[0]          # ← dùng cái này, KHÔNG dùng len()
    n_classes = len(le_stage2.classes_)

    s1_probs = stage1_model.predict_proba(X_test)[:, 1]
    s1_pred  = (s1_probs >= stage1_threshold).astype(int)

    s2_probs = np.zeros((n_test, n_classes))
    mask_s1 = s1_pred == 1
    if mask_s1.sum() > 0:
        s2_probs[mask_s1] = stage2_model.predict_proba(X_test[mask_s1])
    s2_conf = s2_probs.max(axis=1)
    s2_idx  = s2_probs.argmax(axis=1)

    y_pred = np.empty(n_test, dtype=object)
    selected = np.zeros(n_test, dtype=bool)

    for i in range(n_test):
        if s1_pred[i] == 0:
            y_pred[i] = 'Normal'
            selected[i] = True
        elif s2_conf[i] < conf_threshold:
            y_pred[i] = 'Attack_Unknown'
            selected[i] = False
        else:
            y_pred[i] = le_stage2.classes_[s2_idx[i]]
            selected[i] = True

    y_true_sel = y_true[selected]
    y_pred_sel = y_pred[selected]

    present = [c for c in unified_classes
               if c in set(y_true_sel) or c in set(y_pred_sel)]
    sel_mf1 = f1_score(y_true_sel, y_pred_sel,
                       labels=present, average='macro', zero_division=0)
    sel_acc = accuracy_score(y_true_sel, y_pred_sel)

    coverage = selected.sum() / n_test
    attack_mask = y_true != 'Normal'
    attack_covered = (selected & attack_mask).sum() / attack_mask.sum()

    per_class_cov = {}
    for c in unified_classes:
        if c == 'Normal':
            continue
        mask_true = y_true == c
        if mask_true.sum() == 0:
            continue
        per_class_cov[c] = float((selected & mask_true).sum() / mask_true.sum())

    return {
        'stage1_threshold': float(stage1_threshold),
        'conf_threshold':   float(conf_threshold),
        'n_test':           int(n_test),
        'coverage':         float(coverage),
        'attack_coverage':  float(attack_covered),
        'n_selected':       int(selected.sum()),
        'n_unknown':        int((~selected).sum()),
        'selective_macro_f1': float(sel_mf1),
        'selective_accuracy': float(sel_acc),
        'per_class_coverage': per_class_cov,
        'y_pred': y_pred,
        'selected': selected,
        'y_true_sel': y_true_sel,
        'y_pred_sel': y_pred_sel,
    }


def main():
    print("[*] Loading...")
    stage1_model     = joblib.load(MODELS_DIR / "model.pkl")
    stage1_artifacts = load_artifacts(MODELS_DIR)
    stage2_model     = joblib.load(MODELS_DIR / "model_stage2_6class.pkl")
    le_stage2        = joblib.load(MODELS_DIR / "label_encoder_stage2_6class.pkl")

    test_df = drop_leakage(load_raw(TEST_PATH))
    test_df[ATTACK_CAT_COL] = test_df[ATTACK_CAT_COL].astype(str).str.strip()
    X_test, _, _ = transform(test_df, stage1_artifacts)
    n_test = X_test.shape[0]          # ← cache ngay từ đầu

    unified_classes = ['Normal'] + list(le_stage2.classes_)
    y_true = np.array([
        'Normal' if is_atk == 0 else MAP_6CLASS[cat]
        for cat, is_atk in zip(test_df[ATTACK_CAT_COL], test_df[LABEL_COL])
    ])

    # Stage 1 threshold comparison
    print("\n[*] Stage 1 threshold comparison (conf=0.5):")
    for s1_t in [0.5, 0.6, 0.7, 0.75]:
        r = evaluate_selective(
            stage1_model, stage2_model, le_stage2,
            X_test, y_true, unified_classes,
            stage1_threshold=s1_t, conf_threshold=0.5,
        )
        print(f"   s1_t={s1_t:.2f}: coverage={r['coverage']:.3f} | "
              f"attack_cov={r['attack_coverage']:.3f} | "
              f"sel_F1={r['selective_macro_f1']:.4f}")

    S1_THRESHOLD = 0.5
    print(f"\n[*] Sweep conf threshold @ s1_t={S1_THRESHOLD}:")
    rows = []
    for ct in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = evaluate_selective(
            stage1_model, stage2_model, le_stage2,
            X_test, y_true, unified_classes,
            stage1_threshold=S1_THRESHOLD, conf_threshold=ct,
        )
        rows.append({
            'conf_threshold': ct,
            'coverage': r['coverage'],
            'attack_coverage': r['attack_coverage'],
            'selective_macro_f1': r['selective_macro_f1'],
            'n_unknown': r['n_unknown'],
        })
        print(f"   ct={ct:.2f}: coverage={r['coverage']:.3f} | "
              f"attack_cov={r['attack_coverage']:.3f} | "
              f"sel_F1={r['selective_macro_f1']:.4f}")

    valid = [r for r in rows if r['attack_coverage'] >= 0.60]
    best = max(valid, key=lambda r: r['selective_macro_f1']) if valid \
           else max(rows, key=lambda r: r['selective_macro_f1'])
    best_ct = best['conf_threshold']
    print(f"\n[✓] Chọn conf_threshold = {best_ct:.2f} "
          f"(coverage={best['coverage']:.3f}, "
          f"selective_F1={best['selective_macro_f1']:.4f})")

    # Final report
    final = evaluate_selective(
        stage1_model, stage2_model, le_stage2,
        X_test, y_true, unified_classes,
        stage1_threshold=S1_THRESHOLD, conf_threshold=best_ct,
    )

    print(f"\n===== PRODUCTION CONFIG =====")
    print(f"Stage 1 threshold : {S1_THRESHOLD}")
    print(f"Conf threshold    : {best_ct}")
    print(f"Coverage          : {final['coverage']:.4f} "
          f"({final['n_selected']}/{n_test})")     # ← FIX: dùng n_test
    print(f"Attack coverage   : {final['attack_coverage']:.4f}")
    print(f"Selective macro-F1: {final['selective_macro_f1']:.4f}")
    print(f"Selective accuracy: {final['selective_accuracy']:.4f}")

    print(f"\nClassification report (on classified subset):")
    print(classification_report(
        final['y_true_sel'], final['y_pred_sel'],
        labels=[c for c in unified_classes
                if c in set(final['y_true_sel']) | set(final['y_pred_sel'])],
        digits=4, zero_division=0,
    ))

    print(f"\n[*] Per-class coverage (% attacks classified, not Unknown):")
    for c, cov in final['per_class_coverage'].items():
        print(f"   {c:15s}: {cov*100:5.1f}%")

    cm = confusion_matrix(
        final['y_true_sel'], final['y_pred_sel'],
        labels=[c for c in unified_classes
                if c in set(final['y_true_sel']) | set(final['y_pred_sel'])],
    )
    print(f"\nConfusion matrix (classified subset):")
    print(cm)

    # Plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    cts = [r['conf_threshold'] for r in rows]
    f1s = [r['selective_macro_f1'] for r in rows]
    covs = [r['coverage'] for r in rows]
    ax1.plot(cts, f1s, 'o-', color='tab:blue', label='Selective macro-F1')
    ax1.set_xlabel('Confidence threshold')
    ax1.set_ylabel('Selective macro-F1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axvline(best_ct, color='red', linestyle='--',
                label=f'chosen {best_ct:.2f}')
    ax2 = ax1.twinx()
    ax2.plot(cts, covs, 's--', color='tab:orange', label='Coverage')
    ax2.set_ylabel('Coverage', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.title(f'Production: Selective F1 vs Coverage (s1_t={S1_THRESHOLD})')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "selective_6class.png", dpi=150)
    plt.close()
    print(f"\n[✓] Saved selective_6class.png")

    metadata = {
        "task": "hierarchical_6class_production",
        "stage1_threshold": S1_THRESHOLD,
        "conf_threshold":   float(best_ct),
        "coverage":         final['coverage'],
        "attack_coverage":  final['attack_coverage'],
        "selective_macro_f1": final['selective_macro_f1'],
        "selective_accuracy": final['selective_accuracy'],
        "classes": unified_classes,
        "per_class_coverage": final['per_class_coverage'],
    }
    with open(MODELS_DIR / "metadata_6class_production.json", "w") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame(rows).to_csv(
        MODELS_DIR / "selective_sweep_6class.csv", index=False
    )
    print(f"[✓] Saved metadata_6class_production.json")


if __name__ == "__main__":
    main()
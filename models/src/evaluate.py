import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import TEST_PATH, MODELS_DIR, CAT_COLS
from preprocess import load_raw, drop_leakage, load_artifacts, transform


def group_importance(feature_names, importance):
    """Gộp one-hot groups (proto_*, service_*, state_*) thành 1 entry."""
    cat_set = set(CAT_COLS)
    grouped = {}
    for name, imp in zip(feature_names, importance):
        matched = False
        for cat in cat_set:
            if name.startswith(cat + "_"):
                grouped[cat] = grouped.get(cat, 0.0) + float(imp)
                matched = True
                break
        if not matched:
            grouped[name] = grouped.get(name, 0.0) + float(imp)
    return grouped


def main():
    print("[*] Loading model + artifacts...")
    model = joblib.load(MODELS_DIR / "model.pkl")
    artifacts = load_artifacts(MODELS_DIR)

    # Load threshold từ metadata.json (fallback 0.5)
    threshold = 0.5
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        threshold = float(metadata.get("threshold", 0.5))
        print(f"[*] Model: {metadata.get('model_name', '?')} | "
              f"threshold = {threshold:.2f}")
    else:
        print(f"[!] metadata.json not found, using threshold = 0.5")

    test = drop_leakage(load_raw(TEST_PATH))
    X_test, y_test, attack_cat = transform(test, artifacts)

    # Predict bằng threshold đã tune
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        probs = None
        y_pred = model.predict(X_test)

    print("\n===== Overall Metrics (TEST) =====")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Macro-F1 : {f1_score(y_test, y_pred, labels=[0, 1], average='macro', zero_division=0):.4f}")

    if probs is not None and len(np.unique(y_test)) == 2:
        try:
            print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
        except (ValueError, IndexError):
            pass

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4, zero_division=0,
    ))

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("[✓] Saved confusion_matrix.png")

    # ---- Feature importance (grouped) ----
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        feature_names = artifacts["feature_names"]

        if len(importance) == len(feature_names):
            grouped = group_importance(feature_names, importance)
            items = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:20]
            names = [k for k, _ in items]
            vals  = [v for _, v in items]

            plt.figure(figsize=(9, 7))
            sns.barplot(x=vals, y=names)
            plt.title("Top 20 Feature Importances (grouped)")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(MODELS_DIR / "feature_importance.png", dpi=150)
            plt.close()
            print("[✓] Saved feature_importance.png (grouped)")
        else:
            print(f"[!] Feature importance mismatch: "
                  f"{len(importance)} vs {len(feature_names)}")

    # ---- Threshold sweep plot ----
    sweep_path = MODELS_DIR / "threshold_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        plt.figure(figsize=(7, 4))
        plt.plot(sweep["threshold"], sweep["val_macro_f1"], marker="o")
        plt.axvline(threshold, color="red", linestyle="--",
                    label=f"chosen = {threshold:.2f}")
        plt.xlabel("Threshold"); plt.ylabel("Val Macro-F1")
        plt.title("Threshold Sweep (Validation)")
        plt.legend(); plt.tight_layout()
        plt.savefig(MODELS_DIR / "threshold_sweep.png", dpi=150)
        plt.close()
        print("[✓] Saved threshold_sweep.png")

    # ---- Per-attack recall ----
    test_eval = pd.DataFrame({
        "label": y_test,
        "pred": y_pred,
        "attack_cat": np.asarray(attack_cat, dtype=str),
    })
    attacks = test_eval[test_eval["label"] == 1]
    print("\n[*] Recall by attack category:")
    for category, group in attacks.groupby("attack_cat", dropna=False):
        r = (group["pred"] == 1).mean()
        print(f"   {str(category):15s}: {r:.4f} (n={len(group)})")


if __name__ == "__main__":
    main()
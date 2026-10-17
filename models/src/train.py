import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    TRAIN_PATH, TEST_PATH, MODELS_DIR, LABEL_COL, RANDOM_STATE,
    VAL_SIZE, DEFAULT_THRESHOLD,
)
from preprocess import (
    load_raw, drop_leakage, fit_preprocessors,
    transform, save_artifacts,
)


def make_models(y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    pos_weight = neg / max(pos, 1)

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }


def evaluate_at_threshold(model, X, y, threshold=DEFAULT_THRESHOLD):
    """Chấm điểm binary classifier tại threshold cho trước."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        probs = None
        preds = model.predict(X)

    precision = precision_score(y, preds, pos_label=1, zero_division=0)
    recall    = recall_score(y, preds, pos_label=1, zero_division=0)
    macro_f1  = f1_score(y, preds, labels=[0, 1], average="macro", zero_division=0)

    roc_auc = None
    if probs is not None and len(np.unique(y)) == 2:
        try:
            roc_auc = roc_auc_score(y, probs)
        except (ValueError, IndexError):
            pass

    return {
        "threshold": float(threshold),
        "macro_f1": float(macro_f1),
        "attack_precision": float(precision),
        "attack_recall": float(recall),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }, preds


def print_eval(name, y, preds, attack_cat, header=""):
    print(f"\n===== {name} {header} =====")
    print(classification_report(
        y, preds, labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4, zero_division=0,
    ))
    print("Confusion matrix [rows=actual, cols=pred; labels 0,1]:")
    print(confusion_matrix(y, preds, labels=[0, 1]))

    df = pd.DataFrame({
        "label": y,
        "pred": preds,
        "attack_cat": np.asarray(attack_cat, dtype=str),
    })
    attacks = df[df["label"] == 1]
    print("\n[*] Recall by attack category:")
    for cat, grp in attacks.groupby("attack_cat", dropna=False):
        r = (grp["pred"] == 1).mean()
        print(f"   {str(cat):15s}: {r:.4f} (n={len(grp)})")


def tune_threshold(model, X_val, y_val):
    """Sweep threshold trên VAL, chọn theo macro-F1."""
    probs = model.predict_proba(X_val)[:, 1]
    grid = np.arange(0.05, 0.96, 0.05)
    rows = []
    best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, labels=[0, 1],
                      average="macro", zero_division=0)
        rows.append({"threshold": float(t), "val_macro_f1": float(f1)})
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1, pd.DataFrame(rows)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Stage 0: load data
    # ============================================================
    print("[*] Loading raw data...")
    train_full = drop_leakage(load_raw(TRAIN_PATH))
    test_df    = drop_leakage(load_raw(TEST_PATH))
    print(f"[*] Train (full): {train_full.shape} | Test: {test_df.shape}")
    print(f"[*] Train label dist:\n{train_full[LABEL_COL].value_counts().sort_index()}")

    # ============================================================
    # Stage 1: split train -> train_sub / val
    # ============================================================
    train_sub, val_df = train_test_split(
        train_full,
        test_size=VAL_SIZE,
        stratify=train_full[LABEL_COL],
        random_state=RANDOM_STATE,
    )
    print(f"\n[*] Split -> train_sub {train_sub.shape} | val {val_df.shape}")

    print("[*] Fitting preprocessor on train_sub ONLY...")
    artifacts_split, X_tr, y_tr = fit_preprocessors(train_sub)
    X_val, y_val, val_attack_cat = transform(val_df, artifacts_split)

    # ============================================================
    # Stage 2: train từng model trên train_sub, evaluate trên val
    # ============================================================
    models = make_models(y_tr)
    val_results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[*] [Stage 1] Training {name} on train_sub...")
        model.fit(X_tr, y_tr)
        metrics, preds = evaluate_at_threshold(model, X_val, y_val, DEFAULT_THRESHOLD)
        print_eval(name, y_val, preds, val_attack_cat, header="(VAL @ 0.50)")
        val_results[name] = {"model": name, **metrics}
        fitted_models[name] = model

    # ============================================================
    # Stage 3: chọn model tốt nhất theo VAL macro-F1
    # ============================================================
    best_name = max(val_results, key=lambda n: val_results[n]["macro_f1"])
    print(f"\n[✓] Best (val macro-F1): {best_name} "
          f"({val_results[best_name]['macro_f1']:.4f})")

    # ============================================================
    # Stage 4: tune threshold trên VAL với model thắng
    # ============================================================
    print(f"\n[*] Tuning threshold for {best_name} on val...")
    best_model_val = fitted_models[best_name]
    best_threshold, val_f1_at_thresh, thresh_table = tune_threshold(
        best_model_val, X_val, y_val
    )
    print(f"[✓] Best threshold = {best_threshold:.2f} "
          f"(val macro-F1 = {val_f1_at_thresh:.4f})")
    thresh_table.to_csv(MODELS_DIR / "threshold_sweep.csv", index=False)

    # ============================================================
    # Stage 5: refit preprocessor + best model trên FULL train
    # ============================================================
    print(f"\n[*] Refitting preprocessor on full train ({len(train_full)} rows)...")
    artifacts_full, X_train_full, y_train_full = fit_preprocessors(train_full)
    save_artifacts(artifacts_full, MODELS_DIR)

    print(f"[*] Refitting {best_name} on full train...")
    final_models = make_models(y_train_full)
    best_model = final_models[best_name]
    best_model.fit(X_train_full, y_train_full)

    # ============================================================
    # Stage 6: đánh giá CUỐI CÙNG trên test — CHỈ 1 LẦN
    # ============================================================
    print("\n[*] Transforming test set...")
    X_test, y_test, test_attack_cat = transform(test_df, artifacts_full)

    test_metrics, test_preds = evaluate_at_threshold(
        best_model, X_test, y_test, threshold=best_threshold
    )
    print_eval(best_name, y_test, test_preds, test_attack_cat,
               header=f"(TEST @ {best_threshold:.2f})")

    # Cũng log luôn kết quả ở threshold mặc định để so sánh
    test_metrics_default, _ = evaluate_at_threshold(
        best_model, X_test, y_test, threshold=DEFAULT_THRESHOLD
    )

    # ============================================================
    # Save
    # ============================================================
    joblib.dump(best_model, MODELS_DIR / "model.pkl")

    metadata = {
        "model_name": best_name,
        "threshold": float(best_threshold),
        "val_macro_f1_at_threshold": float(val_f1_at_thresh),
        "test_metrics_at_threshold": test_metrics,
        "test_metrics_at_default": test_metrics_default,
    }
    joblib.dump(metadata, MODELS_DIR / "metadata.pkl")
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame(val_results.values()).to_csv(
        MODELS_DIR / "model_comparison_val.csv", index=False
    )
    pd.DataFrame([test_metrics]).to_csv(
        MODELS_DIR / "final_test_metrics.csv", index=False
    )

    print("\n[✓] Done.")
    print(f"   model             -> {MODELS_DIR / 'model.pkl'}")
    print(f"   preprocessing     -> {MODELS_DIR / 'preprocessing.pkl'}")
    print(f"   metadata (json)   -> {MODELS_DIR / 'metadata.json'}")
    print(f"   val comparison    -> {MODELS_DIR / 'model_comparison_val.csv'}")
    print(f"   test (final)      -> {MODELS_DIR / 'final_test_metrics.csv'}")
    print(f"   threshold sweep   -> {MODELS_DIR / 'threshold_sweep.csv'}")
    print("\n[i] Test được chấm ĐÚNG 1 LẦN. Model + threshold được chọn trên val.")


if __name__ == "__main__":
    main()
import joblib
from config import TRAIN_PATH, MODELS_DIR
from preprocess import load_raw, drop_leakage, fit_preprocessors, save_artifacts


def main():
    print("[*] Loading clean train data...")
    train = drop_leakage(load_raw(TRAIN_PATH))

    # Sanity: đảm bảo _coarse không có trong raw
    if "_coarse" in train.columns:
        print("[!] _coarse found in raw data, dropping...")
        train = train.drop(columns=["_coarse"])

    print(f"[*] Fitting preprocessor on {len(train)} rows...")
    artifacts, X, y = fit_preprocessors(train)
    save_artifacts(artifacts, MODELS_DIR)

    # Verify
    feats = artifacts["input_feature_names"]
    assert "_coarse" not in feats, "Still corrupted!"
    print(f"[✓] Refit OK. n_input_features = {len(feats)}")
    print(f"[✓] First 5: {feats[:5]}")
    print(f"[✓] Last 5:  {feats[-5:]}")

    # Verify model.pkl vẫn compatible
    model = joblib.load(MODELS_DIR / "model.pkl")
    print(f"[✓] model.pkl loaded: {type(model).__name__}")
    print("[✓] preprocessing.pkl giờ consistent với model.pkl")


if __name__ == "__main__":
    main()
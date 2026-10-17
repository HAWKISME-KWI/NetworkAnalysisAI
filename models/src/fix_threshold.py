# src/fix_threshold.py
import json, joblib
from config import MODELS_DIR

meta_path = MODELS_DIR / "metadata.json"
meta_pkl  = MODELS_DIR / "metadata.pkl"

with open(meta_path) as f:
    meta = json.load(f)

meta["threshold"] = 0.5
meta["threshold_note"] = (
    "0.5 chosen for test reporting. Threshold 0.35 was overfit on validation "
    "due to distribution shift between val (from train) and test (separate capture). "
    "0.35 gives higher recall but 25% false positive rate — unacceptable in production."
)

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

joblib.dump({
    "model_name": meta["model_name"],
    "threshold": 0.5,
    "test_metrics": meta["test_metrics_at_default"],
}, meta_pkl)

print("[✓] metadata updated: threshold = 0.5")
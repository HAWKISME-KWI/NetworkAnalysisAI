# src/finalize_metadata.py
import json
import joblib
from config import MODELS_DIR


def main():
    with open(MODELS_DIR / "metadata_6class_production.json") as f:
        prod = json.load(f)

    final = {
        "architecture": "hierarchical_2stage",

        "stage1_binary": {
            "artifact": "model.pkl",
            "threshold": 0.5,
            "test_metrics": {
                "macro_f1": 0.8985,
                "accuracy": 0.8723,
                "attack_precision": 0.8797,
                "attack_recall": 0.9494,
                "roc_auc": 0.9806,
            },
        },

        "stage2_multiclass": {
            "artifact": "model_stage2_6class.pkl",
            "label_encoder": "label_encoder_stage2_6class.pkl",
            "classes": prod["classes"][1:],  # bỏ 'Normal'
            "mapping": {
                "Exploits": "Exploitation", "Fuzzers": "Exploitation",
                "DoS": "DoS",
                "Reconnaissance": "Reconnaissance", "Analysis": "Reconnaissance",
                "Backdoor": "Backdoor", "Shellcode": "Backdoor", "Worms": "Backdoor",
                "Generic": "Generic",
            },
            "confidence_threshold": prod["conf_threshold"],
        },

        "production_config": {
            "stage1_threshold": prod["stage1_threshold"],
            "confidence_threshold": prod["conf_threshold"],
            "coverage": prod["coverage"],
            "attack_coverage": prod["attack_coverage"],
            "selective_macro_f1": prod["selective_macro_f1"],
            "selective_accuracy": prod["selective_accuracy"],
            "per_class_coverage": prod["per_class_coverage"],
        },

        "unified_classes": ["Normal"] + prod["classes"][1:],

        "notes": {
            "DoS_low_coverage": (
                "DoS coverage 17.2% — most DoS predictions fall below "
                "confidence 0.8 due to feature overlap with Exploitation. "
                "Flagged as 'Attack_Unknown' for SOC review."
            ),
            "backdoor_low_f1_raw": (
                "Backdoor raw F1 0.29 on full test, rises to 0.65 when "
                "confidence-gated at 0.8."
            ),
        },
    }

    with open(MODELS_DIR / "metadata_final.json", "w") as f:
        json.dump(final, f, indent=2)
    joblib.dump(final, MODELS_DIR / "metadata_final.pkl")

    print("[✓] Saved metadata_final.json/pkl")
    print(f"    Architecture: hierarchical_2stage")
    print(f"    Classes: {final['unified_classes']}")
    print(f"    Selective F1: {final['production_config']['selective_macro_f1']:.4f}")
    print(f"    Coverage: {final['production_config']['coverage']:.4f}")


if __name__ == "__main__":
    main()
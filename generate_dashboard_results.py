import os
import json
import time

BASE = os.path.dirname(os.path.abspath(__file__))

CNN_FILE = os.path.join(BASE, "models", "results", "cnn", "training_results.json")
ENS_FILE = os.path.join(BASE, "models", "results", "ensemble", "ensemble_report.json")

OUT_DIR = os.path.join(BASE, "models", "results", "dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "dashboard_results.json")

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_format(model_obj):
    """
    Konversi semua model agar memiliki struktur standar:
    {
      "model": "...",
      "accuracy": 0.xx,
      "f1_macro": 0.xx,
      "precision_macro": 0.xx,
      "recall_macro": 0.xx,
      "confusion_matrix": [...],
      "classification_report": {...},
      "history": {...},
      "created_at": "..."
    }
    """
    if not isinstance(model_obj, dict):
        return {}

    std = {}
    for key in [
        "model", "accuracy", "f1_macro", "precision_macro",
        "recall_macro", "confusion_matrix", "classification_report",
        "history", "created_at"
    ]:
        std[key] = model_obj.get(key)

    # fallback
    if std["created_at"] is None:
        std["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return std


def main():
    cnn_data = load_json(CNN_FILE)
    ens_data = load_json(ENS_FILE)

    final = {
        "cnn": {},
        "ensemble": {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # === CNN (list → dict)
    if isinstance(cnn_data, list):
        for item in cnn_data:
            key = item.get("model") or f"cnn_{len(final['cnn'])+1}"
            final["cnn"][key] = ensure_format(item)

    # === ENSEMBLE (results inside)
    if isinstance(ens_data, dict):
        res = ens_data.get("results", {})
        if isinstance(res, dict):
            for key, item in res.items():
                final["ensemble"][key] = ensure_format(item)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("SUCCESS: dashboard_results.json generated")
    print("Location:", OUT_FILE)


if __name__ == "__main__":
    main()

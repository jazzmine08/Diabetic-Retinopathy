import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# -----------------------------------------------------------
#  LOAD DATASET (real data from processed_final/)
# -----------------------------------------------------------
def load_dataset(test_dir):
    X = []
    y = []

    for label in sorted(os.listdir(test_dir)):
        folder = os.path.join(test_dir, label)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            try:
                img = load_img(img_path, target_size=(224, 224))
                arr = img_to_array(img) / 255.0
                X.append(arr)
                y.append(int(label))
            except Exception:
                continue

    X = np.array(X)
    y = np.array(y)

    return X, y


# -----------------------------------------------------------
#  PLOT ROC SAVER
# -----------------------------------------------------------
def save_roc_curve(model_name, y_true, y_prob, out_dir):
    fpr = {}
    tpr = {}
    roc_auc = {}

    num_classes = y_prob.shape[1]
    y_true_onehot = np.eye(num_classes)[y_true]

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Save JSON raw data
    roc_json_path = os.path.join(out_dir, f"{model_name}_roc.json")
    with open(roc_json_path, "w") as f:
        json.dump({
            "fpr": all_fpr.tolist(),
            "tpr": mean_tpr.tolist(),
            "auc": float(macro_auc)
        }, f, indent=2)

    # Save ROC PNG
    plt.figure(figsize=(6, 6))
    plt.plot(all_fpr, mean_tpr, label=f"AUC = {macro_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{model_name}_roc.png"))
    plt.close()

    return macro_auc


# -----------------------------------------------------------
#  CONFUSION MATRIX SAVER
# -----------------------------------------------------------
def save_confusion_matrix(model_name, y_true, y_pred, out_dir, labels):
    cm = confusion_matrix(y_true, y_pred)

    # Save JSON raw
    with open(os.path.join(out_dir, f"{model_name}_cm.json"), "w") as f:
        json.dump({
            "labels": labels,
            "matrix": cm.tolist()
        }, f, indent=2)

    # Save PNG image
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.savefig(os.path.join(out_dir, f"{model_name}_cm.png"))
    plt.close()


# -----------------------------------------------------------
#  EVALUATOR FOR CNN (.h5)
# -----------------------------------------------------------
def evaluate_cnn(model_path, X, y):
    model = load_model(model_path)

    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = (y_pred == y).mean() * 100

    return y_pred, y_prob, acc


# -----------------------------------------------------------
#  EVALUATOR FOR ENSEMBLE (.pkl)
# -----------------------------------------------------------
def evaluate_ensemble(model_path, X, y):
    clf = joblib.load(model_path)

    # reshape images to 1D
    X_flat = X.reshape(len(X), -1)

    y_pred = clf.predict(X_flat)

    # some models have predict_proba, some do not
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_flat)
    else:
        # fallback: one-hot fake probability
        num_classes = len(np.unique(y))
        y_prob = np.eye(num_classes)[y_pred]

    acc = (y_pred == y).mean() * 100

    return y_pred, y_prob, acc


# -----------------------------------------------------------
#  MAIN EVALUATION FUNCTION (dipanggil dari app.py)
# -----------------------------------------------------------
def evaluate_all_models(test_dir, model_paths, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset(test_dir)
    labels = sorted(list({int(c) for c in y}))

    summary = {}

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            summary[model_name] = {"error": "Model file not found", "path": model_path}
            continue

        print(f"\nEvaluating {model_name} ...")

        # CNN
        if model_path.endswith(".h5"):
            y_pred, y_prob, acc = evaluate_cnn(model_path, X, y)

        # Ensemble
        elif model_path.endswith(".pkl"):
            y_pred, y_prob, acc = evaluate_ensemble(model_path, X, y)

        # Unknown format
        else:
            summary[model_name] = {"error": "Unknown model format", "path": model_path}
            continue

        # Save ROC & Confusion Matrix
        auc_score = save_roc_curve(model_name, y, y_prob, out_dir)
        save_confusion_matrix(model_name, y, y_pred, out_dir, labels)

        # Save single report
        report = classification_report(y, y_pred, output_dict=True)

        summary[model_name] = {
            "accuracy": float(acc),
            "auc": float(auc_score),
            "report": report,
            "roc_json": f"{model_name}_roc.json",
            "roc_png": f"{model_name}_roc.png",
            "cm_json": f"{model_name}_cm.json",
            "cm_png": f"{model_name}_cm.png"
        }

    # save global summary
    with open(os.path.join(out_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary

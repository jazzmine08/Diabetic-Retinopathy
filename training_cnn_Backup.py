# training_cnn.py
"""
Training CNN pipeline — saves models to models/cnn/ and results to models/results/cnn/
Emits socket events:
 - training_log (progress messages)
 - training_epoch (per-epoch summary)
 - training_batch (per-batch progress, optional)
 - training_done (finished)
 - training_error (on exception)
"""

import os
import re
import time
import json
import traceback
import unicodedata
import shutil

# Non-GUI matplotlib backend pre-config
import matplotlib
matplotlib.use("Agg")
os.environ["TK_SILENCE_DEPRECATION"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import (
    DenseNet121,
    InceptionResNetV2,
    EfficientNetV2B0
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import classification_report, confusion_matrix

AUTOTUNE = tf.data.experimental.AUTOTUNE


# =========================================================
# Helper: safe filename normalization
# =========================================================
def _safe_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name or "file"


def _normalize_and_clean_dir(root_dir: str, socketio=None):
    renamed, removed, total_checked = [], [], 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            total_checked += 1
            abs_path = os.path.join(dirpath, fname)
            try:
                try:
                    size = os.path.getsize(abs_path)
                except Exception:
                    size = -1

                if size == 0:
                    os.remove(abs_path)
                    removed.append((abs_path, "zero_size"))
                    continue

                safe = _safe_name(fname)
                if safe != fname:
                    new_path = os.path.join(dirpath, safe)
                    base, ext = os.path.splitext(safe)
                    i = 1
                    while os.path.exists(new_path):
                        new_path = os.path.join(dirpath, f"{base}_{i}{ext}")
                        i += 1
                    try:
                        os.rename(abs_path, new_path)
                        renamed.append((abs_path, new_path))
                        abs_path = new_path
                    except Exception:
                        try:
                            shutil.copy2(abs_path, new_path)
                            os.remove(abs_path)
                            renamed.append((abs_path, new_path))
                            abs_path = new_path
                        except Exception:
                            pass

            except Exception as e:
                removed.append((abs_path, f"error:{e}"))

    if socketio:
        if renamed:
            socketio.emit("training_log", {"message": f"🔁 Renamed {len(renamed)} files (normalized)."})
        if removed:
            socketio.emit("training_log", {"message": f"🗑️ Removed {len(removed)} problematic files."})

    return {"renamed": renamed, "removed": removed, "checked": total_checked}


# =========================================================
# Callback class
# =========================================================
class SocketProgressCallback(Callback):
    def __init__(self, socketio, model_name, epochs, steps_per_epoch, model_index=0, total_models=1):
        super().__init__()
        self.socketio = socketio
        self.model_name = model_name
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch or 1)
        self.model_index = model_index
        self.total_models = total_models
        self._epoch_start_time = None

    def on_train_begin(self, logs=None):
        self.socketio.emit("training_log", {"message":
            f"▶️ Mulai training {self.model_name} ({self.model_index+1}/{self.total_models})"})

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()
        self.socketio.emit("training_log", {"message":
            f"⏳ Epoch {epoch+1}/{self.epochs} untuk {self.model_name}"})

    def on_train_batch_end(self, batch, logs=None):
        try:
            percent = int(((batch + 1) / self.steps_per_epoch) * 100)
            self.socketio.emit("training_batch", {
                "model": self.model_name,
                "epoch": (self.params.get("epoch", 0) + 1),
                "batch": batch + 1,
                "batch_progress": percent,
            })
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        try:
            t = time.time() - self._epoch_start_time
            train_acc = logs.get("accuracy")
            val_acc = logs.get("val_accuracy")
            self.socketio.emit("training_epoch", {
                "model": self.model_name,
                "epoch": epoch + 1,
                "epochs": self.epochs,
                "train_acc": float(train_acc * 100) if train_acc else None,
                "val_acc": float(val_acc * 100) if val_acc else None,
                "epoch_time_s": round(t, 2)
            })
        except Exception:
            pass


# =========================================================
# Build model head
# =========================================================
def _build_head(base, num_classes):
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs=base.input, outputs=out)


# =========================================================
# Create dataset safely
# =========================================================
def _make_datasets(processed_dir, image_size, batch_size, cache, seed, socketio):
    socketio.emit("training_log", {"message": "🧹 Normalisasi file dataset..."})
    _normalize_and_clean_dir(processed_dir, socketio=socketio)

    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        socketio.emit("training_log", {"message": "📂 Menggunakan struktur train/val."})
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, labels="inferred", label_mode="categorical",
            image_size=image_size, batch_size=batch_size,
            shuffle=True, seed=seed)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir, labels="inferred", label_mode="categorical",
            image_size=image_size, batch_size=batch_size,
            shuffle=False)
    else:
        socketio.emit("training_log", {"message": "📂 Split internal 80/20."})
        train_ds = tf.keras.utils.image_dataset_from_directory(
            processed_dir, labels="inferred", label_mode="categorical",
            image_size=image_size, batch_size=batch_size,
            shuffle=True, validation_split=0.2,
            subset="training", seed=seed)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            processed_dir, labels="inferred", label_mode="categorical",
            image_size=image_size, batch_size=batch_size,
            shuffle=False, validation_split=0.2,
            subset="validation", seed=seed)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    class_names = train_ds.class_names
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())

    socketio.emit("training_log", {"message": f"✅ Dataset siap — {len(class_names)} kelas."})
    return train_ds, val_ds, len(class_names), steps_per_epoch


# =========================================================
# Evaluate and Save Results (Confusion + Report)
# =========================================================
def evaluate_and_save(model, val_ds, model_name, results_dir):
    """Evaluasi dan simpan confusion matrix + classification report"""
    y_true, y_pred = [], []

    for batch_images, batch_labels in val_ds:
        preds = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))

    cr = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    eval_path = os.path.join(results_dir, f"{model_name}_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump({
            "model": model_name,
            "accuracy": cr.get("accuracy", 0),
            "classification_report": cr,
            "confusion_matrix": cm
        }, f, indent=2)

    return eval_path


# =========================================================
# Main Training Function
# =========================================================
def train_all_cnn(socketio,
                  processed_dir,
                  models_dir,
                  results_dir,
                  epochs=5,
                  batch_size=16,
                  cache_dataset=False,
                  image_size=(224,224)):

    try:
        socketio.emit("training_log", {"message": "🚀 Training CNN — pipeline aktif."})

        # Create folders
        models_cnn_dir = os.path.join(models_dir, "cnn")
        results_cnn_dir = os.path.join(results_dir, "cnn")
        os.makedirs(models_cnn_dir, exist_ok=True)
        os.makedirs(results_cnn_dir, exist_ok=True)

        # Dataset
        train_ds, val_ds, num_classes, steps_per_epoch = _make_datasets(
            processed_dir, image_size, batch_size, cache_dataset, 123, socketio)

        # Define models
        backbones = {
            "DenseNet121": DenseNet121(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3)),
            "InceptionResNetV2": InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3)),
            "EfficientNetV2": EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3))
        }

        results = []

        # Train each model
        for idx, (name, backbone) in enumerate(backbones.items(), start=1):
            try:
                socketio.emit("training_log", {"message": f"🔧 Menyiapkan {name} ({idx}/{len(backbones)})"})

                model = _build_head(backbone, num_classes)
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])

                save_path = os.path.join(models_cnn_dir, f"{name}.keras")
                checkpoint = ModelCheckpoint(save_path, monitor="val_accuracy",
                                             save_best_only=True, mode="max", verbose=1)

                cb_socket = SocketProgressCallback(socketio, name, epochs, steps_per_epoch,
                                                   model_index=idx-1, total_models=len(backbones))

                t0 = time.time()
                history = model.fit(train_ds, validation_data=val_ds,
                                    epochs=epochs, callbacks=[checkpoint, cb_socket],
                                    verbose=0)
                elapsed = time.time() - t0

                best_val = max(history.history.get("val_accuracy", [0]))

                socketio.emit("training_log", {"message": f"📊 Evaluasi {name}..."})
                eval_path = evaluate_and_save(model, val_ds, name, results_cnn_dir)

                results.append({
                    "model": name,
                    "best_val_accuracy": float(best_val),
                    "train_time_s": round(elapsed, 2),
                    "model_path": save_path,
                    "evaluation_path": eval_path,
                    "history": {k: [float(x) for x in v] for k, v in history.history.items()}
                })

            except Exception as e:
                socketio.emit("training_log", {"message": f"❌ Error model {name}: {e}"})

        # Save overall result summary
        summary_path = os.path.join(results_cnn_dir, "training_results.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        socketio.emit("training_done", {"status": "ok", "results_path": summary_path})
        return results

    except Exception as e:
        tb = traceback.format_exc()
        socketio.emit("training_error", {"message": str(e), "trace": tb})
        raise

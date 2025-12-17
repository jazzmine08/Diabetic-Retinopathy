# training_cnn.py
"""
Training CNN pipeline — saves models to models/cnn/ and results to models/results/cnn/
Outputs JSON in "Format A" per model (see spec).
Emits socket events for UI.
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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

AUTOTUNE = tf.data.experimental.AUTOTUNE

# -------------------------
# Helpers
# -------------------------
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
        for fname in list(filenames):
            total_checked += 1
            abs_path = os.path.join(dirpath, fname)
            try:
                try:
                    size = os.path.getsize(abs_path)
                except Exception:
                    size = -1
                if size == 0:
                    try:
                        os.remove(abs_path)
                        removed.append((abs_path, "zero_size"))
                    except Exception:
                        removed.append((abs_path, "unremovable_zero"))
                    continue

                safe = _safe_name(fname)
                if safe != fname:
                    new_path = os.path.join(dirpath, safe)
                    i = 1
                    base, ext = os.path.splitext(safe)
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
                if not os.path.exists(abs_path):
                    removed.append((abs_path, "missing_after_rename"))
            except Exception as e:
                removed.append((abs_path, f"error:{e}"))

    if socketio:
        if renamed:
            socketio.emit("training_log", {"message": f"🔁 Renamed {len(renamed)} files (normalized)."})
            for old, new in renamed[:20]:
                socketio.emit("training_log", {"message": f"    {os.path.relpath(old)} -> {os.path.relpath(new)}"})
        if removed:
            socketio.emit("training_log", {"message": f"🗑️ Removed {len(removed)} problematic files."})

    return {"renamed": renamed, "removed": removed, "checked": total_checked}

# -------------------------
# Callback
# -------------------------
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
        try:
            self.socketio.emit("training_log", {"message": f"▶️ Mulai training {self.model_name} ({self.model_index+1}/{self.total_models})"})
        except Exception:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()
        try:
            self.socketio.emit("training_log", {"message": f"⏳ Epoch {epoch+1}/{self.epochs} untuk {self.model_name}"})
        except Exception:
            pass

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
            t = time.time() - (self._epoch_start_time or time.time())
            train_acc = logs.get("accuracy")
            val_acc = logs.get("val_accuracy")
            self.socketio.emit("training_epoch", {
                "model": self.model_name,
                "epoch": epoch + 1,
                "epochs": self.epochs,
                "train_acc": float(train_acc * 100) if train_acc is not None else None,
                "val_acc": float(val_acc * 100) if val_acc is not None else None,
                "epoch_time_s": round(t, 2)
            })
        except Exception:
            pass

# -------------------------
# Build head
# -------------------------
def _build_head(base, num_classes):
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs=base.input, outputs=out)

# -------------------------
# Dataset maker
# -------------------------
def _make_datasets(processed_dir, image_size, batch_size, cache, seed, socketio):
    if socketio:
        socketio.emit("training_log", {"message": "🧹 Normalisasi file dataset..."})
    _normalize_and_clean_dir(processed_dir, socketio=socketio)

    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")

    # STEP 1 — Load raw dataset first (without prefetch)
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        if socketio:
            socketio.emit("training_log", {"message": f"📂 Menggunakan struktur train/val di {processed_dir}"})

        raw_train = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed
        )
        raw_val = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False
        )

    else:
        if socketio:
            socketio.emit("training_log", {"message": f"📂 Menggunakan struktur kelas di {processed_dir} (split internal 80/20)"})

        raw_train = tf.keras.utils.image_dataset_from_directory(
            processed_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            subset="training",
            seed=seed
        )
        raw_val = tf.keras.utils.image_dataset_from_directory(
            processed_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            validation_split=0.2,
            subset="validation",
            seed=seed
        )

    # STEP 2 — SAFE: class_names before prefetch
    class_names = raw_train.class_names
    num_classes = len(class_names)

    # STEP 3 — now apply cache/prefetch
    if cache:
        train_ds = raw_train.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = raw_val.cache().prefetch(buffer_size=AUTOTUNE)
    else:
        train_ds = raw_train.prefetch(buffer_size=AUTOTUNE)
        val_ds = raw_val.prefetch(buffer_size=AUTOTUNE)

    try:
        steps_per_epoch = int(tf.data.experimental.cardinality(raw_train).numpy())
    except:
        steps_per_epoch = None

    if socketio:
        socketio.emit("training_log", {
            "message": f"✅ Dataset siap — {num_classes} kelas, steps_per_epoch: {steps_per_epoch}"
        })

    return train_ds, val_ds, num_classes, steps_per_epoch


# -------------------------
# Evaluate & save (Format A)
# -------------------------
def evaluate_and_save(model, val_ds, model_name, results_dir):
    y_true, y_pred = [], []
    for batch_images, batch_labels in val_ds:
        preds = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    if len(y_true) == 0:
        cr = {}
        cm = []
        acc = 0.0
        p_macro = r_macro = f_macro = 0.0
        num_classes = 0
    else:
        cr = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        num_classes = int(max(max(y_true), max(y_pred)) + 1) if y_true else 0

    eval_path = os.path.join(results_dir, f"{model_name}_evaluation.json")
    payload = {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": float(f_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "num_classes": int(num_classes),
        "classification_report": cr,
        "confusion_matrix": cm,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(eval_path, "w") as f:
        json.dump(payload, f, indent=2)
    return eval_path, payload

# -------------------------
# Main training function (public)
# -------------------------
def train_all_cnn(socketio,
                  processed_dir,
                  models_dir,
                  results_dir,
                  epochs=5,
                  batch_size=16,
                  cache_dataset=False,
                  image_size=(224,224)):
    """
    Saves models to: <models_dir>/cnn/<ModelName>.keras
    Saves results to: <results_dir>/cnn/training_results.json
    Evaluation per-model to: <results_dir>/cnn/<ModelName>_evaluation.json
    Outputs JSON in Format A (shown earlier).
    """
    try:
        socketio.emit("training_log", {"message": "🚀 Memulai Training CNN (3 model) — pipeline."})

        # enable mixed precision if GPU present (safe try)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                socketio.emit("training_log", {"message": "⚡ GPU detected — mixed precision ON."})
        except Exception:
            pass

        models_cnn_dir = os.path.join(models_dir, "cnn")
        results_cnn_dir = os.path.join(results_dir, "cnn")
        os.makedirs(models_cnn_dir, exist_ok=True)
        os.makedirs(results_cnn_dir, exist_ok=True)

        train_ds, val_ds, num_classes, steps_per_epoch = _make_datasets(
            processed_dir, image_size, batch_size, cache_dataset, 123, socketio)

        if num_classes is None or num_classes == 0:
            raise RuntimeError("Tidak ada kelas ditemukan di dataset.")

        backbones = {
            "DenseNet121": DenseNet121(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3)),
            "InceptionResNetV2": InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3)),
            "EfficientNetV2": EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3))
        }

        results = []
        for idx, (name, backbone) in enumerate(backbones.items(), start=1):
            try:
                socketio.emit("training_log", {"message": f"🔧 Menyiapkan model {name} ({idx}/{len(backbones)})"})
                model = _build_head(backbone, num_classes)
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                              loss="categorical_crossentropy", metrics=["accuracy"])

                save_path = os.path.join(models_cnn_dir, f"{name}.keras")
                checkpoint = ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

                cb_socket = SocketProgressCallback(socketio, name, epochs, steps_per_epoch or 1,
                                                   model_index=idx-1, total_models=len(backbones))

                t0 = time.time()
                history = model.fit(train_ds, validation_data=val_ds,
                                    epochs=epochs, callbacks=[checkpoint, cb_socket],
                                    verbose=0)
                elapsed = time.time() - t0

                best_val = float(max(history.history.get("val_accuracy", [0.0])))

                # run evaluation and save in Format A
                socketio.emit("training_log", {"message": f"📊 Evaluasi {name}... (saving results)"})
                eval_path, eval_payload = evaluate_and_save(model, val_ds, name, results_cnn_dir)

                # build model record (Format A)
                record = {
                    "model": name,
                    "accuracy": eval_payload.get("accuracy", 0.0),
                    "f1_macro": eval_payload.get("f1_macro", 0.0),
                    "precision_macro": eval_payload.get("precision_macro", 0.0),
                    "recall_macro": eval_payload.get("recall_macro", 0.0),
                    "num_classes": eval_payload.get("num_classes", num_classes),
                    "classification_report": eval_payload.get("classification_report", {}),
                    "confusion_matrix": eval_payload.get("confusion_matrix", []),
                    "best_val_accuracy": best_val,
                    "train_time_s": round(elapsed, 2),
                    "model_path": save_path,
                    "evaluation_path": eval_path,
                    "history": {k: [float(x) for x in v] for k, v in history.history.items()}
                }
                results.append(record)

                socketio.emit("training_log", {"message": f"✅ Selesai {name} — best_val_acc: {best_val:.4f} — waktu: {elapsed:.1f}s"})

            except Exception as e_mod:
                tb = traceback.format_exc()
                socketio.emit("training_log", {"message": f"❌ Gagal training {name}: {e_mod}"})
                socketio.emit("training_log", {"message": tb})
                # continue to next model

        # Save summary (array of model records)
        summary_path = os.path.join(results_cnn_dir, "training_results.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        socketio.emit("training_done", {"status": "ok", "results_path": summary_path, "results": results})

        # ==========================================
        # SAVE GLOBAL TRAINING RESULTS (dashboard)
        # ==========================================
        training_summary = []

        for item in results:
            training_summary.append({
                "model": item.get("model"),
                "best_val_accuracy": item.get("best_val_accuracy"),
                "train_time_s": item.get("train_time_s"),
                "history": item.get("history"),
                "created_at": item.get("created_at")
            })

        save_file = os.path.join(results_cnn_dir, "training_results.json")

        with open(save_file, "w") as f:
            json.dump(training_summary, f, indent=2)

        socketio.emit("training_done", {
            "message": "Training CNN selesai — training_results.json dibuat."
        })

       
       
       
        return results

    except Exception as e:
        tb = traceback.format_exc()
        try:
            socketio.emit("training_error", {"message": str(e), "trace": tb})
        except Exception:
            pass
        raise

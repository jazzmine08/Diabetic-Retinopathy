import os
import time
import shutil
import base64
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from threading import Thread

# =========================================================
# KONFIGURASI DASAR
# =========================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_final")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# ROUTING HALAMAN (TIDAK ADA *_page LAGI)
# =========================================================
@app.route("/", endpoint="index")
def index():
    # contoh data dummy agar dashboard tidak error
    return render_template(
        "index.html",
        chart_data={
            "labels": ["CNN", "VGG16", "ResNet50", "InceptionV3", "Ensemble"],
            "scores": [86, 89, 91, 90, 93],
        },
        best_model={"model": "Ensemble CNN", "accuracy": 93, "f1": 92},
        dataset_stats={"train": 2500, "val": 400, "test": 300},
    )

@app.route("/preprocessing", endpoint="preprocessing")
def preprocessing():
    return render_template("preprocessing.html")

@app.route("/eda", endpoint="eda")
def eda():
    return render_template("eda.html")

@app.route("/training", endpoint="training")
def training():
    return render_template("training.html")

@app.route("/ensemble", endpoint="ensemble")
def ensemble():
    return render_template("ensemble.html")

@app.route("/hasil_model", endpoint="hasil_model")
def hasil_model():
    return render_template("hasil_model.html")


# =========================================================
# PREPROCESSING — CLEAN + FULL FEATURE (REALTIME PREVIEW + POPUP)
# =========================================================
@app.route("/start_preprocessing", methods=["POST"])
def start_preprocessing():
    clear = request.args.get("clear", "false").lower() == "true"
    fast = request.args.get("fast", "false").lower() == "true"
    socketio.start_background_task(run_preprocessing_worker, clear=clear, fast=fast)
    return jsonify({"status": "started", "message": "Preprocessing dimulai..."})


def run_preprocessing_worker(clear=False, fast=False):
    try:
        socketio.emit("preprocess_log", {"message": "🟢 Preprocessing dimulai..."})

        # folder raw harus ada
        if not os.path.exists(RAW_DIR):
            socketio.emit("preprocess_error", {"message": f"Folder {RAW_DIR} tidak ditemukan"})
            return

        # bersihkan folder jika clear=True
        if clear:
            shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            socketio.emit("preprocess_log", {"message": "🧹 Folder lama dibersihkan."})

        # list file
        items = []
        for root, _, files in os.walk(RAW_DIR):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    absf = os.path.join(root, f)
                    relf = os.path.relpath(absf, RAW_DIR)
                    items.append((absf, relf))

        total = len(items)
        if total == 0:
            socketio.emit("preprocess_error", {"message": "❌ Tidak ada gambar di folder raw/."})
            return

        socketio.emit("preprocess_log", {"message": f"📂 {total} gambar ditemukan."})

        processed = 0
        MAX_PREVIEWS = 12
        start_time = time.time()

        for absf, relf in items:
            img = cv2.imread(absf)
            if img is None:
                continue

            # cropping retina
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                img = img[y:y+h, x:x+w]

            # CLAHE
            if not fast:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l2 = clahe.apply(l)
                img = cv2.cvtColor(cv2.merge((l2,a,b)), cv2.COLOR_LAB2BGR)

            img = cv2.resize(img, (224,224))

            out_dir = os.path.join(PROCESSED_DIR, os.path.dirname(relf))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(relf)), img)

            processed += 1
            progress = int(processed/total*100)
            elapsed = time.time()-start_time
            eta = int((elapsed/max(1,processed))*(total-processed))

            socketio.emit("preprocess_update", {
                "progress": progress,
                "message": f"{processed}/{total} selesai • ETA {eta}s"
            })

            # kirim preview max 12
            if processed <= MAX_PREVIEWS:
                pv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pv = cv2.resize(pv, (128,128))
                _, buf = cv2.imencode(".jpg", pv)
                socketio.emit("preprocess_preview", {
                    "image": base64.b64encode(buf).decode("utf-8")
                })
                socketio.sleep(0.1)

        total_time = time.time()-start_time
        socketio.emit("preprocess_done", {"message": f"Selesai {processed}/{total} gambar • {total_time:.1f}s"})
        socketio.emit("show_popup", {"message": "Preprocessing selesai!"})

    except Exception as e:
        socketio.emit("preprocess_error", {"message": f"❌ Kesalahan: {e}"})
# =========================================================
# ===============  E D A   F U N G S I   ==================
# =========================================================
import hashlib
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter

def analyze_dataset(base_dir):
    """
    EDA dataset hasil preprocessing (folder 0–4)
    Menghasilkan:
      - distribusi per kelas
      - file corrupt
      - file duplikat
      - statistik intensitas / saturasi / kontras
      - histogram kelas (base64)
      - contoh gambar tiap kelas (max 2)
    """

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Folder tidak ditemukan: {base_dir}")

    class_counts = Counter()
    corrupted = []
    hashes = Counter()

    pixel_intensity = []
    pixel_saturation = []
    pixel_contrast = []

    # ===============================
    # LOOP FOLDER KELAS
    # ===============================
    for cls_name in sorted(os.listdir(base_dir), key=lambda x: str(x)):
        cls_path = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue

        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fname in files:
            fpath = os.path.join(cls_path, fname)

            try:
                img = cv2.imread(fpath)
                if img is None:
                    corrupted.append(fpath)
                    continue

                # cek hash untuk duplikat
                with open(fpath, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                hashes[h] += 1

                # statistik piksel
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                pixel_intensity.append(np.mean(gray))
                pixel_saturation.append(np.mean(hsv[:, :, 1]))
                pixel_contrast.append(np.std(gray))

                class_counts[cls_name] += 1

            except Exception:
                corrupted.append(fpath)

    # jumlah duplikat
    duplicates = sum(c - 1 for c in hashes.values() if c > 1)

    # ===============================
    # HISTOGRAM DISTRIBUSI KELAS
    # ===============================
    fig, ax = plt.subplots()
    ax.bar(class_counts.keys(), class_counts.values(), color="skyblue")
    ax.set_title("Distribusi Data per Kelas (0–4)")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Gambar")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    histogram_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # ===============================
    # PREVIEW GAMBAR PER KELAS
    # ===============================
    sample_images = []
    for cls_name in sorted(class_counts.keys(), key=lambda x: str(x)):
        cls_path = os.path.join(base_dir, cls_name)
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))][:2]

        for fname in imgs:
            fp = os.path.join(cls_path, fname)
            try:
                img = Image.open(fp)
                img.thumbnail((160, 160))

                buf = BytesIO()
                img.save(buf, format="JPEG")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

                sample_images.append({
                    "class": cls_name,
                    "img": encoded
                })
            except:
                continue

    # ===============================
    # RETURN HASIL AKHIR EDA
    # ===============================
    return {
        "status": "ok",
        "total_files": sum(class_counts.values()),
        "class_distribution": dict(class_counts),
        "corrupted": corrupted,
        "duplicates": duplicates,
        "mean_intensity": float(np.mean(pixel_intensity)) if pixel_intensity else 0,
        "mean_saturation": float(np.mean(pixel_saturation)) if pixel_saturation else 0,
        "mean_contrast": float(np.mean(pixel_contrast)) if pixel_contrast else 0,
        "histogram": histogram_b64,
        "sample_images": sample_images
    }


# =========================================================
# =============== ROUTE: JALANKAN EDA =====================
# =========================================================
@app.route("/run_eda", methods=["GET"])
def run_eda():
    try:
        result = analyze_dataset(PROCESSED_DIR)
        return jsonify(result)
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print("=== EDA ERROR ===")
        print(trace)
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": trace
        }), 500
# =========================================================
# ===============   T R A I N I N G   =====================
# =========================================================

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, InceptionResNetV2, EfficientNetV2B0
from tensorflow.keras import layers, models

def build_model(base, n_classes=5):
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


@app.route("/start_training", methods=["POST"])
def start_training():
    socketio.start_background_task(training_worker)
    return jsonify({"status": "started", "message": "Training dimulai..."})


def training_worker():
    try:
        socketio.emit("training_log", {"message": "🚀 Mulai training 3 model CNN..."})

        # Dataset generator
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

        train_gen = datagen.flow_from_directory(
            PROCESSED_DIR,
            target_size=(224, 224),
            class_mode="categorical",
            batch_size=16,
            subset="training"
        )

        val_gen = datagen.flow_from_directory(
            PROCESSED_DIR,
            target_size=(224, 224),
            class_mode="categorical",
            batch_size=16,
            subset="validation"
        )

        n_classes = train_gen.num_classes

        MODELS = {
            "DenseNet121": DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3)),
            "InceptionResNetV2": InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3)),
            "EfficientNetV2B0": EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(224,224,3))
        }

        results = {}

        for model_name, base_model in MODELS.items():

            socketio.emit("training_log", {"message": f"📦 Melatih {model_name}..."})
            model = build_model(base_model, n_classes)

            hist = model.fit(
                train_gen,
                epochs=5,
                validation_data=val_gen,
                verbose=0,
                callbacks=[
                    TrainingProgress(model_name)
                ]
            )

            # Simpan model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            model.save(model_path)

            socketio.emit("training_log", {"message": f"💾 Model disimpan: {model_path}"})

            # Simpan hasil training
            results[model_name] = {
                "train_acc": float(hist.history["accuracy"][-1]),
                "val_acc": float(hist.history["val_accuracy"][-1]),
                "train_loss": float(hist.history["loss"][-1]),
                "val_loss": float(hist.history["val_loss"][-1])
            }

        socketio.emit("training_done", {
            "message": "🎉 Training semua model selesai!",
            "results": results
        })

    except Exception as e:
        socketio.emit("training_error", {"message": f"❌ Training gagal: {e}"})


# =========================================================
# SOCKET.IO CALLBACK PROGRESS
# =========================================================
class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.progress = 0

    def on_epoch_end(self, epoch, logs=None):
        self.progress += 20
        socketio.emit("training_progress", {
            "model": self.model_name,
            "progress": self.progress,
            "acc": float(logs.get("accuracy", 0)),
            "val_acc": float(logs.get("val_accuracy", 0)),
        })

import os
import time
import base64
import hashlib
import traceback
from io import BytesIO

import cv2
import numpy as np
from split_dataset import split_dataset


# ============================================================
# Helper Functions
# ============================================================

def _b64_from_image(img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _ensure_subdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _is_image(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def _safe_filename(name):
    """Remove illegal characters & force .jpg extension"""
    base, ext = os.path.splitext(name)
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        ext = ".jpg"

    # remove illegal chars
    safe = "".join(c for c in base if c.isalnum() or c in ("-", "_"))
    if safe == "":
        safe = "image"

    return safe + ext


def _hash_image(img):
    """Create hash to detect duplicates"""
    try:
        return hashlib.md5(img.tobytes()).hexdigest()
    except:
        return None


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def run_preprocessing(socketio, clear_data, fast_mode, raw_dir, processed_dir):
    """
    Master preprocessing function with:
    - Safe rename
    - Auto-fix missing ext
    - Duplicate detection
    - Corrupt file detection
    - Heavy logging
    - Automatic dataset split
    """

    try:
        socketio.emit("preprocess_log", {"message": "🟢 Preprocessing dimulai..."})

        if not os.path.exists(raw_dir):
            socketio.emit("preprocess_error", {"message": f"❌ RAW folder tidak ditemukan: {raw_dir}"})
            return

        _ensure_subdir(processed_dir)

        # --------------------------------------------------------
        # CLEAR OLD DATA
        # --------------------------------------------------------
        if clear_data:
            socketio.emit("preprocess_log", {"message": "🧹 Menghapus hasil preprocessing sebelumnya..."})
            for root, dirs, files in os.walk(processed_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except:
                        pass

        # --------------------------------------------------------
        # SCAN FILES + SAFE RENAME
        # --------------------------------------------------------
        items = []
        rename_map = {}
        existing_files = set()

        socketio.emit("preprocess_log", {"message": "🔍 Memindai file RAW..."})

        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                old_path = os.path.join(root, f)

                # --- FIX missing extension ---
                if "." not in f:
                    f += ".jpg"

                new_name = _safe_filename(f)

                # --- FIX duplicate file names ---
                while new_name in existing_files:
                    base, ext = os.path.splitext(new_name)
                    new_name = f"{base}_dup{ext}"

                existing_files.add(new_name)

                new_path = os.path.join(root, new_name)
                rename_map[old_path] = new_path

        # rename safely
        for old, new in rename_map.items():
            if old != new:
                try:
                    os.rename(old, new)
                    socketio.emit("preprocess_log", {
                        "message": f"♻️ Rename aman: {os.path.basename(old)} → {os.path.basename(new)}"
                    })
                except:
                    pass

        # --------------------------------------------------------
        # RE-SCAN after rename
        # --------------------------------------------------------
        items = []
        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                if _is_image(f):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, raw_dir)
                    items.append((abs_path, rel_path))

        total = len(items)
        if total == 0:
            socketio.emit("preprocess_error", {"message": "❌ Tidak ada gambar ditemukan setelah rename!"})
            return

        socketio.emit("preprocess_log", {"message": f"📂 Total {total} gambar siap diproses."})

        # --------------------------------------------------------
        # Duplicate detection via hashing
        # --------------------------------------------------------
        socketio.emit("preprocess_log", {"message": "🧬 Deteksi duplikat gambar..."})
        seen_hash = set()

        # --------------------------------------------------------
        # PROCESS LOOP
        # --------------------------------------------------------
        processed = 0
        start_time = time.time()
        MAX_PREVIEW = 12
        preview_count = 0

        for idx, (abs_path, rel_path) in enumerate(items):

            try:
                img = cv2.imread(abs_path)

                # ---- Fix Corrupt File ----
                if img is None:
                    socketio.emit("preprocess_log", {
                        "message": f"⚠️ Corrupt / tidak bisa dibaca: {abs_path} → dilewati"
                    })
                    continue

                # ---- Duplicate Detection ----
                h = _hash_image(img)
                if h in seen_hash:
                    socketio.emit("preprocess_log", {
                        "message": f"⚠️ Duplikat terdeteksi: {rel_path} → dilewati"
                    })
                    continue
                seen_hash.add(h)

                # ================================
                # 1. Crop by FOV
                # ================================
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        x, y, w, h = cv2.boundingRect(max(cnts, key=lambda c: cv2.contourArea(c)))
                        img = img[y:y+h, x:x+w]
                except:
                    pass

                # ================================
                # 2. Illumination Correction
                # ================================
                try:
                    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=10)
                    img = cv2.addWeighted(img, 1.0, blur, -0.1, 0)
                except:
                    pass

                # ================================
                # 3. Gray World Balancing
                # ================================
                try:
                    arr = img.astype(np.float32)
                    avg = arr.mean(axis=(0, 1))
                    gain = avg.mean() / np.where(avg == 0, 1, avg)
                    img = np.clip(arr * gain, 0, 255).astype(np.uint8)
                except:
                    pass

                # ================================
                # 4. CLAHE (skip if fast mode)
                # ================================
                if not fast_mode:
                    try:
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
                    except:
                        pass

                # ================================
                # 5. Vessel Sharpening
                # ================================
                try:
                    kernel = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]])
                    img = cv2.filter2D(img, -1, kernel)
                except:
                    pass

                # ================================
                # 6. Resize & Save
                # ================================
                img_out = cv2.resize(img, (224, 224))
                out_folder = os.path.join(processed_dir, os.path.dirname(rel_path))
                _ensure_subdir(out_folder)
                cv2.imwrite(os.path.join(out_folder, os.path.basename(rel_path)), img_out)

                # ================================
                # Preview (first 12)
                # ================================
                if preview_count < MAX_PREVIEW:
                    b64 = _b64_from_image(cv2.resize(img_out, (128, 128)))
                    if b64:
                        socketio.emit("preprocess_preview", {"image": b64})
                    preview_count += 1

                # ================================
                # Progress Update
                # ================================
                processed += 1
                progress = int(processed / total * 100)
                elapsed = time.time() - start_time
                eta = int((elapsed / processed) * (total - processed))

                socketio.emit("preprocess_update", {
                    "progress": progress,
                    "message": f"{processed}/{total} selesai",
                    "eta": eta
                })

            except Exception as e:
                socketio.emit("preprocess_log", {
                    "message": f"⚠️ ERROR file {rel_path}: {e}"
                })

        # selesai
        total_time = time.time() - start_time
        socketio.emit("preprocess_done", {
            "message": f"✅ Preprocessing selesai: {processed}/{total} file dalam {total_time:.1f}s"
        })

        # --------------------------------------------------------
        # AUTO SPLIT DATASET
        # --------------------------------------------------------
        socketio.emit("preprocess_log", {"message": "📤 Split dataset otomatis..."})

        training_output = os.path.join(os.path.dirname(processed_dir), "training")

        split_dataset(
            processed_dir=processed_dir,
            output_dir=training_output
        )

        socketio.emit("preprocess_log", {
            "message": "✔ Dataset telah di-split menjadi train/val/test"
        })

    except Exception as e:
        socketio.emit("preprocess_error", {
            "message": f"❌ ERROR Fatal: {e}",
            "trace": traceback.format_exc()
        })

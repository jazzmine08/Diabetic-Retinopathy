# ============================================================
# eda.py FINAL — FULLY COMPATIBLE WITH FRONTEND JS
# ============================================================

import os
import cv2
import json
import numpy as np
import base64
import traceback
from io import BytesIO
from matplotlib import pyplot as plt


# ============================================================
# Helper: Convert image → Base64
# ============================================================

def img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


# ============================================================
# Helper: Generate histogram → Base64
# ============================================================

def generate_histogram(image_list):
    """Generate a global histogram from sample images."""
    if len(image_list) == 0:
        return None

    all_pixels = np.concatenate([img.flatten() for img in image_list])

    plt.figure(figsize=(5, 3))
    plt.hist(all_pixels, bins=50)
    plt.title("Global Intensity Histogram")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# Main EDA Function (Dipanggil oleh Flask)
# ============================================================

def run_eda_analysis(
    dataset_dir="dataset/raw"
):
    """
    Melakukan EDA dan mengembalikan dictionary untuk dikirim sebagai JSON.
    """

    results = {
        "total_files": 0,
        "num_classes": 0,
        "mean_intensity": None,
        "mean_saturation": None,
        "mean_contrast": None,
        "histogram": None,
        "corrupted_files": [],
        "duplicate_files": [],
        "sample_images": {},
        "pixel_stats": {}
    }

    try:
        if not os.path.exists(dataset_dir):
            return {"status": "error", "message": f"Folder {dataset_dir} tidak ditemukan"}

        classes = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])

        results["num_classes"] = len(classes)

        all_images = []
        hashes = {}
        duplicated = []

        mean_intensity_list = []
        mean_sat_list = []
        contrast_list = []

        sample_images = {}

        # ============================================================
        # LOOP PER KELAS
        # ============================================================

        for cls in classes:
            cls_path = os.path.join(dataset_dir, cls)
            files = [
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
            ]

            results["total_files"] += len(files)
            sample_images[cls] = []

            for i, fn in enumerate(files):
                fpath = os.path.join(cls_path, fn)

                # ----------------------------------------------------
                # Baca image
                # ----------------------------------------------------
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        results["corrupted_files"].append(fpath)
                        continue
                except:
                    results["corrupted_files"].append(fpath)
                    continue

                # ----------------------------------------------------
                # Duplicate detection via hashing
                # ----------------------------------------------------
                h = hash(img.tobytes())
                if h in hashes:
                    duplicated.append(fpath)
                else:
                    hashes[h] = fpath

                # ----------------------------------------------------
                # Pixel stats
                # ----------------------------------------------------
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                mean_intensity_list.append(np.mean(gray))
                mean_sat_list.append(np.mean(hsv[:, :, 1]))

                # simple local contrast
                contrast_list.append(np.std(gray))

                # ----------------------------------------------------
                # Ambil sample (max 3)
                # ----------------------------------------------------
                if len(sample_images[cls]) < 3:
                    sample_images[cls].append(img_to_base64(img))

                # kumpulkan pixel untuk histogram
                if len(all_images) < 10:  
                    all_images.append(gray)

        # ============================================================
        # FINALIZE METRICS
        # ============================================================

        results["mean_intensity"] = float(np.mean(mean_intensity_list)) if mean_intensity_list else 0
        results["mean_saturation"] = float(np.mean(mean_sat_list)) if mean_sat_list else 0
        results["mean_contrast"] = float(np.mean(contrast_list)) if contrast_list else 0

        results["histogram"] = generate_histogram(all_images)
        results["duplicate_files"] = duplicated
        results["sample_images"] = sample_images

        results["pixel_stats"] = {
            "mean_intensity_per_image": float(np.mean(mean_intensity_list)) if mean_intensity_list else 0,
            "std_intensity_per_image": float(np.std(mean_intensity_list)) if mean_intensity_list else 0,
            "mean_saturation_per_image": float(np.mean(mean_sat_list)) if mean_sat_list else 0,
            "mean_contrast_per_image": float(np.mean(contrast_list)) if contrast_list else 0,
        }

        return results

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

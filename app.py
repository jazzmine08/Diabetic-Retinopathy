# ============================================================
# app.py FINAL (Merged, cleaned, full-dashboard on /hasil_model)
# ============================================================

import os
import json
import traceback
import threading
import subprocess
import sys
import csv
import io
from flask import (
    Flask, render_template, jsonify, request,
    send_from_directory, send_file, current_app
)
from flask_socketio import SocketIO

# ============================================================
# IMPORT USER MODULES (optional; failures are tolerated)
# ============================================================

# Preprocessing
try:
    from preprocessing import run_preprocessing
except Exception as e:
    print("❌ preprocessing.py error:", e)
    run_preprocessing = None

# Training CNN
try:
    from training_cnn import train_all_cnn
except Exception as e:
    print("❌ training_cnn.py error:", e)
    train_all_cnn = None

# Ensemble
try:
    from ensemble import run_ensemble
except Exception as e:
    print("❌ ensemble.py error:", e)
    run_ensemble = None

# Evaluation (optional)
try:
    from evaluation import evaluate_all_models
except Exception as e:
    print("❌ evaluation.py error:", e)
    evaluate_all_models = None


# ============================================================
# PATH CONFIG (GLOBAL FOLDER)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed_final")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "models", "results")
TEST_DIR = os.path.join(BASE_DIR, "test_final")

# ensure directories exist
for p in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, TEST_DIR]:
    os.makedirs(p, exist_ok=True)

# canonical file locations used by dashboard & APIs
TRAINING_RESULT_FILE = os.path.join(RESULTS_DIR, "cnn", "training_results.json")
ENSMBLE_RESULT_FILE = os.path.join(RESULTS_DIR, "ensemble", "ensemble_report.json")
EVAL_RESULT_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")

# ensure subfolders exist
os.makedirs(os.path.join(RESULTS_DIR, "cnn"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "ensemble"), exist_ok=True)


# ============================================================
# INIT FLASK + SOCKETIO
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "secretkey123"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ============================================================
# ROUTES (HALAMAN UTAMA)
# ============================================================

@app.route("/")
def index():

    def count_images(path):
        total = 0
        if not os.path.exists(path):
            return 0
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    total += 1
        return total

    dataset_stats = {
        "train": count_images(os.path.join(PROCESSED_DIR, "train")),
        "val": count_images(os.path.join(PROCESSED_DIR, "val")),
        "test": count_images(os.path.join(PROCESSED_DIR, "test"))
    }

    # best model (from training results list)
    best_model = None
    if os.path.exists(TRAINING_RESULT_FILE):
        try:
            with open(TRAINING_RESULT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            if isinstance(results, list) and len(results) > 0:
                best = max(results, key=lambda x: x.get("accuracy", 0))
                best_model = {
                    "model": best.get("model", "Unknown"),
                    "accuracy": round(best.get("accuracy", 0) * 100, 2) if best.get("accuracy") is not None else None,
                    "f1": round(best.get("f1_score", 0) * 100, 2) if best.get("f1_score") is not None else None
                }
        except Exception as e:
            print("Error reading training results:", e)

    # chart_data for index (simple)
    chart_data = {"labels": [], "scores": []}
    if os.path.exists(TRAINING_RESULT_FILE):
        try:
            with open(TRAINING_RESULT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            if isinstance(results, list):
                for entry in results:
                    chart_data["labels"].append(entry.get("model", ""))
                    value = (
                        entry.get("accuracy") or
                        entry.get("acc") or
                        entry.get("val_accuracy") or
                        entry.get("val_acc") or
                        0
                    )
                    try:
                        chart_data["scores"].append(round(float(value) * 100, 2))
                    except:
                        chart_data["scores"].append(0)
        except Exception as e:
            print("Error preparing chart data:", e)

    return render_template(
        "index.html",
        dataset_stats=dataset_stats,
        best_model=best_model,
        chart_data=chart_data
    )


# ============================================================
# EDA / PREPROCESSING / TRAINING / ENSEMBLE ROUTES
# ============================================================

@app.route("/eda")
def eda():
    return render_template("eda.html")

@app.route("/run_eda", methods=["GET"])
def run_eda():
    try:
        from eda import run_eda_analysis
        results = run_eda_analysis()
        return jsonify({"status": "ok", **results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/training")
def training_cnn():
    return render_template("training_cnn.html")


@app.route("/start_training", methods=["POST"])
def start_training():
    if train_all_cnn is None:
        return jsonify({"status": "error", "message": "training_cnn.py tidak tersedia"}), 500
    try:
        socketio.start_background_task(
            train_all_cnn,
            socketio,
            PROCESSED_DIR,
            MODELS_DIR,
            RESULTS_DIR
        )
        return jsonify({"status": "ok", "message": "Training CNN dimulai"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


@app.route("/ensemble")
def ensemble():
    return render_template("ensemble.html")


@app.route("/start_ensemble", methods=["POST"])
def start_ensemble():
    if run_ensemble is None:
        return jsonify({"status": "error", "message": "ensemble.py tidak tersedia"}), 500
    try:
        # optional params from client
        data = request.get_json(silent=True) or {}
        force_recompute = bool(data.get("force_recompute", False))
        use_optuna = bool(data.get("use_optuna", False))

        # run ensemble in background
        socketio.start_background_task(
            run_ensemble,
            socketio,
            MODELS_DIR,
            RESULTS_DIR,
            TEST_DIR,
            force_recompute,
            use_optuna
        )
        return jsonify({"status":"ok","message":"Ensemble pipeline started"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


@app.route("/get_ensemble_results", methods=["GET"])
def get_ensemble_results():
    path = os.path.join(RESULTS_DIR, "ensemble_model", "results_summary.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return jsonify({"status":"ok", "results": json.load(f)})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)}), 500
    return jsonify({"status":"empty", "results": None})


# ============================================================
#   ENDPOINT GENERATE HASIL MODEL (VALIDATOR)
#   -> tidak menjalankan script eksternal; hanya validasi file JSON
# ============================================================

@app.route("/generate_results", methods=["GET"])
def generate_results():
    cnn_path = os.path.join(RESULTS_DIR, "cnn", "training_results.json")
    ens_path = os.path.join(RESULTS_DIR, "ensemble", "ensemble_report.json")

    missing = []
    if not os.path.exists(cnn_path):
        missing.append(cnn_path)
    if not os.path.exists(ens_path):
        missing.append(ens_path)

    if missing:
        return jsonify({
            "status": "error",
            "message": "File hasil model tidak ditemukan",
            "missing_files": missing
        }), 404

    # validate JSON
    try:
        with open(cnn_path, "r", encoding="utf-8") as f:
            json.load(f)
        with open(ens_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "JSON tidak valid",
            "detail": str(e)
        }), 500

    return jsonify({
        "status": "ok",
        "message": "Semua hasil model sudah siap dibaca dashboard"
    })


# ============================================================
#  VIEW: hasil_model (FULL DASHBOARD) — menggantikan dashboard.html
# ============================================================

@app.route("/hasil_model", methods=["GET"])
def hasil_model():
    # render the full dashboard template (hasil_model.html)
    return render_template("hasil_model.html")


# ============================================================
#  API: Gabungkan CNN + Ensemble -> dipakai frontend
# ============================================================

@app.route("/api/evaluation", methods=["GET"])
def api_evaluation():
    combined = {"cnn": {}, "ensemble": {}, "generated_at": None}

    # load cnn results (expected list -> convert to dict keyed by model)
    cnn_path = os.path.join(RESULTS_DIR, "cnn", "training_results.json")
    if os.path.exists(cnn_path):
        try:
            with open(cnn_path, "r", encoding="utf-8") as f:
                cnn_raw = json.load(f)
            if isinstance(cnn_raw, list):
                for i, item in enumerate(cnn_raw):
                    key = item.get("model") or f"cnn_{i+1}"
                    combined["cnn"][key] = item
            elif isinstance(cnn_raw, dict):
                combined["cnn"] = cnn_raw
        except Exception as e:
            print("Error reading CNN JSON:", e)

    # load ensemble
    ens_path = os.path.join(RESULTS_DIR, "ensemble", "ensemble_report.json")
    if os.path.exists(ens_path):
        try:
            with open(ens_path, "r", encoding="utf-8") as f:
                ens_raw = json.load(f)
            if isinstance(ens_raw, dict):
                combined["ensemble"] = ens_raw.get("results", {}) or {}
        except Exception as e:
            print("Error reading ensemble JSON:", e)

    # generated_at (use evaluation file if present)
    if os.path.exists(EVAL_RESULT_FILE):
        try:
            with open(EVAL_RESULT_FILE, "r", encoding="utf-8") as f:
                ev = json.load(f)
                combined["generated_at"] = ev.get("generated_at") or ev.get("created_at") or None
        except:
            combined["generated_at"] = None
    else:
        combined["generated_at"] = None

    return jsonify({"status": "ok", "data": combined})


# ============================================================
# EVALUATE MODELS (wrapper if evaluation module available)
# ============================================================

@app.route("/evaluate_models", methods=["POST"])
def evaluate_models_route():
    if evaluate_all_models is None:
        return jsonify({"status": "error", "message": "evaluation.py tidak ditemukan"}), 500

    try:
        results = evaluate_all_models(
            test_dir=PROCESSED_DIR,
            model_paths={},  # auto_load
            out_dir=RESULTS_DIR
        )

        with open(EVAL_RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return jsonify({"status": "ok", "results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# DOWNLOAD FILES
# ============================================================

@app.route("/results/<path:filename>")
def download_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/models/<path:filename>")
def download_models(filename):
    return send_from_directory(MODELS_DIR, filename)


# ============================================================
# UTILS: get_model_results / export_results_csv
# ============================================================

@app.route("/get_model_results")
def get_model_results():
    results_dir = os.path.join(BASE_DIR, "models")
    results_data = {}
    try:
        for model_file in os.listdir(results_dir):
            if model_file.endswith(".json"):
                model_name = model_file.replace(".json", "")
                file_path = os.path.join(results_dir, model_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        results_data[model_name] = data
                except Exception as e:
                    print(f"Error reading {model_file}: {e}")
    except Exception as e:
        print("get_model_results error:", e)

    return jsonify({"models": results_data})


@app.route("/export_results_csv")
def export_results_csv():
    eval_file = EVAL_RESULT_FILE
    train_file = TRAINING_RESULT_FILE

    rows = []
    headers = ["model", "accuracy", "f1_macro", "notes"]

    # evaluation preferred
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                ev = json.load(f)
            results = ev.get("results") if isinstance(ev, dict) else ev
            if isinstance(results, dict):
                for m, r in results.items():
                    rows.append([m, r.get("accuracy"), r.get("f1_macro"), "evaluation"])
        except Exception as e:
            print("export_results_csv (evaluation) error:", e)

    # fallback training results
    if not rows and os.path.exists(train_file):
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                tr = json.load(f)
            if isinstance(tr, list):
                for t in tr:
                    name = t.get("model", t.get("name"))
                    acc = t.get("best_val_accuracy") or t.get("accuracy")
                    rows.append([name, acc, "", "training"])
        except Exception as e:
            print("export_results_csv (training) error:", e)

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(headers)
    for r in rows:
        cw.writerow(r)

    mem = io.BytesIO(si.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="ensemble_results.csv")


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    print("🚀 Server berjalan di http://127.0.0.1:8000")
    socketio.run(app, host="127.0.0.1", port=8000, debug=True, use_reloader=False)


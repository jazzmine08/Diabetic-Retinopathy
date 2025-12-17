# ensemble.py
"""
Ensemble pipeline (MLP + RandomForest + XGBoost) using:
 - image-level features (hist/hog/haralick) and/or
 - CNN softmax-probs from models stored under models/cnn/

Saves:
 - ensemble models -> <models_dir>/ensemble/*.pkl
 - ensemble results -> <results_dir>/ensemble/ensemble_report.json

Function signature:
  run_ensemble(socketio, processed_dir, models_dir, results_dir, use_cnn, use_image_feats, n_pca)
"""

import os
import time
import json
import traceback
from pathlib import Path

import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# image features
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops

# xgboost import optional
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# tensorflow for loading cnn models
import tensorflow as tf

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def _emit(socketio, event, payload):
    try:
        socketio.emit(event, payload)
    except Exception:
        print("Socket emit failed:", event, payload)

def _list_images_in_dir(root_dir):
    imgs = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(ALLOWED_EXT):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)

def load_cnn_models(models_cnn_dir):
    models = {}
    p = Path(models_cnn_dir)
    if not p.exists():
        return models
    for f in p.iterdir():
        if f.suffix.lower() in (".keras", ".h5", ".hdf5"):
            name = f.stem
            try:
                m = tf.keras.models.load_model(str(f), compile=False)
                models[name] = m
            except Exception as e:
                _emit(None, "ensemble_log", {"message": f"⚠️ Failed to load {f.name}: {e}"})
                try:
                    m = tf.keras.models.load_model(str(f))
                    models[name] = m
                except Exception as e2:
                    print("Failed to load", f, e2)
    return models

def predict_cnn_probs(models_dict, image_paths, image_size=(224,224), batch=32):
    if not models_dict:
        return {}
    n = len(image_paths)
    imgs = np.zeros((n, image_size[0], image_size[1], 3), dtype=np.float32)
    for i, pth in enumerate(image_paths):
        im = cv2.imread(pth)
        if im is None:
            imgs[i] = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)
            continue
        img = cv2.resize(im, (image_size[1], image_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if img.shape != imgs[i].shape:
            img = cv2.resize(img, (image_size[1], image_size[0]))
        imgs[i] = img
    results = {}
    for name, model in models_dict.items():
        try:
            preds = model.predict(imgs, batch_size=batch, verbose=0)
            if preds.ndim == 1:
                preds = np.expand_dims(preds, axis=1)
            results[name] = preds
        except Exception as e:
            print("CNN predict error for", name, e)
    return results

# Image features
def feat_hist_rgb(img, bins_per_channel=32):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = img
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([rgb], [ch], None, [bins_per_channel], [0,256]).flatten()
        hist = hist / (hist.sum()+1e-9)
        feats.append(hist)
    return np.concatenate(feats)

def feat_haralick(img_gray, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    arr = np.clip((img_gray * 255), 0, 255).astype(np.uint8) if img_gray.dtype==np.float32 else img_gray
    glcm = graycomatrix(arr, distances=distances, angles=angles, symmetric=True, normed=True)
    props = []
    for prop in ("contrast","dissimilarity","homogeneity","energy","correlation","ASM"):
        try:
            vals = graycoprops(glcm, prop)
            props.append(float(np.mean(vals)))
            props.append(float(np.std(vals)))
        except Exception:
            props.append(0.0); props.append(0.0)
    return np.array(props)

def feat_hog_single(img_gray, pixels_per_cell=(16,16)):
    try:
        fd = hog(img_gray, pixels_per_cell=pixels_per_cell, cells_per_block=(2,2), feature_vector=True)
        return fd
    except Exception:
        small = cv2.resize((img_gray*255).astype(np.uint8), (32,32)).flatten() / 255.0
        return small

def extract_image_features(image_paths, bins=32, hog_ppc=(16,16)):
    feats = []
    for p in image_paths:
        im = cv2.imread(p)
        if im is None:
            h = np.zeros(bins*3 + 12 + 128)
            feats.append(h); continue
        hist = feat_hist_rgb(im, bins)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        har = feat_haralick(gray)
        hogf = feat_hog_single(gray_f, pixels_per_cell=hog_ppc)
        vec = np.concatenate([hist, har, hogf])
        feats.append(vec)
    maxlen = max(v.shape[0] for v in feats) if feats else 0
    X = np.zeros((len(feats), maxlen), dtype=np.float32)
    for i,v in enumerate(feats):
        X[i,:v.shape[0]] = v
    return X

def _gather_paths_labels(base_dir):
    res = {}
    for split in ("train","val","test"):
        d = os.path.join(base_dir, split)
        if not os.path.isdir(d):
            res[split] = ([], [], [])
            continue
        class_names = sorted([x for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))])
        paths = []; labels = []
        for ci,cls in enumerate(class_names):
            clsdir = os.path.join(d, cls)
            for root, _, files in os.walk(clsdir):
                for f in files:
                    if f.lower().endswith(ALLOWED_EXT):
                        paths.append(os.path.join(root, f))
                        labels.append(int(cls) if cls.isdigit() else cls)
        res[split] = (paths, labels, class_names)
    return res

def run_ensemble(socketio, processed_dir, models_dir, results_dir, use_cnn=True, use_image_feats=True, n_pca=150):
    """
    socketio: SocketIO instance (for emits)
    processed_dir: path to dataset/training (train/val/test/... subfolders)
    models_dir: root models dir (e.g. "models") -> will use models/cnn and models/ensemble
    results_dir: root results dir (e.g. "models/results") -> will use results/ensemble
    """
    try:
        _emit(socketio, "ensemble_log", {"message": "🔍 Mulai pipeline Ensemble (menggunakan data real)."})
        # ensure folders
        models_cnn_dir = os.path.join(models_dir, "cnn")
        models_ensemble_dir = os.path.join(models_dir, "ensemble")
        results_ensemble_dir = os.path.join(results_dir, "ensemble")
        os.makedirs(models_cnn_dir, exist_ok=True)
        os.makedirs(models_ensemble_dir, exist_ok=True)
        os.makedirs(results_ensemble_dir, exist_ok=True)

        splits = _gather_paths_labels(processed_dir)
        train_paths, train_labels, class_names = splits.get("train", ([], [], []))
        val_paths, val_labels, _ = splits.get("val", ([], [], []))
        test_paths, test_labels, _ = splits.get("test", ([], [], []))

        if len(train_paths) == 0 and len(val_paths) == 0:
            raise RuntimeError(f"No images found under {processed_dir}/train and /val. Aborting ensemble.")

        if len(test_paths) == 0:
            _emit(socketio, "ensemble_log", {"message": "⚠️ Warning: No test images found — evaluation will use validation set."})

        cnn_models = load_cnn_models(models_cnn_dir) if use_cnn else {}
        if cnn_models:
            _emit(socketio, "ensemble_log", {"message": f"✅ Ditemukan CNN models: {list(cnn_models.keys())}"})
        else:
            _emit(socketio, "ensemble_log", {"message": "ℹ️ Tidak ada CNN models ditemukan — akan pakai hanya fitur gambar."})

        def prepare_features(paths, split_name):
            _emit(socketio, "ensemble_log", {"message": f"🔧 Ekstraksi fitur: {split_name} ({len(paths)} images)"} )
            feats_list = []
            if use_image_feats:
                X_img = extract_image_features(paths, bins=32, hog_ppc=(16,16))
                feats_list.append(X_img)
                _emit(socketio, "ensemble_log", {"message": f"   • image-feats shape: {X_img.shape}"})
            if use_cnn and cnn_models:
                cnn_preds_all = predict_cnn_probs(cnn_models, paths, image_size=(224,224), batch=32)
                cnn_names = sorted(list(cnn_preds_all.keys()))
                if cnn_names:
                    cnn_concat = np.concatenate([cnn_preds_all[n] for n in cnn_names], axis=1)
                    feats_list.append(cnn_concat)
                    _emit(socketio, "ensemble_log", {"message": f"   • cnn-probs shape: {cnn_concat.shape}"})
            if not feats_list:
                raise RuntimeError("No features extracted (both use_image_feats and use_cnn are False).")
            X_comb = np.concatenate(feats_list, axis=1)
            return X_comb

        X_train = prepare_features(train_paths, "train") if train_paths else np.empty((0,0))
        y_train = np.array(train_labels) if train_labels else np.array([])

        X_val = prepare_features(val_paths, "val") if val_paths else np.empty((0,0))
        y_val = np.array(val_labels) if val_labels else np.array([])

        if test_paths:
            X_test = prepare_features(test_paths, "test")
            y_test = np.array(test_labels)
            test_for_eval = "test"
        else:
            X_test = X_val.copy() if X_val.size else np.empty((0,0))
            y_test = y_val.copy() if y_val.size else np.array([])
            test_for_eval = "val"

        _emit(socketio, "ensemble_log", {"message": f"📦 Feature shapes — train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}"})

        # PCA
        random_state = 42
        if n_pca and n_pca > 0 and X_train.shape[1] > n_pca:
            _emit(socketio, "ensemble_log", {"message": f"📉 Menjalankan PCA (n_components={n_pca})"})
            pca = PCA(n_components=min(n_pca, X_train.shape[1]), random_state=random_state)
            pca.fit(X_train)
            X_train_p = pca.transform(X_train)
            X_val_p = pca.transform(X_val) if X_val.size else X_val
            X_test_p = pca.transform(X_test) if X_test.size else X_test
        else:
            pca = None
            X_train_p, X_val_p, X_test_p = X_train, X_val, X_test

        _emit(socketio, "ensemble_log", {"message": f"✅ PCA done. shapes now: {X_train_p.shape}, {X_val_p.shape}, {X_test_p.shape}"})

        if X_train_p.shape[0] == 0:
            raise RuntimeError("Empty training features — cannot train ensemble.")

        if X_val_p.shape[0] == 0:
            _emit(socketio, "ensemble_log", {"message": "⚠️ Val set kosong. Membuat holdout dari train (20%)."})
            X_train_p, X_val_p, y_train, y_val = train_test_split(X_train_p, y_train, test_size=0.2, random_state=random_state)

        classes_unique = np.unique(y_train)
        uniq_sorted = sorted(classes_unique.tolist()) if classes_unique.size else []
        lbl_map = {v:i for i,v in enumerate(uniq_sorted)}
        def _map(y):
            return np.array([lbl_map[v] for v in y]) if y.size else np.array([])
        y_train_mapped = _map(y_train)
        y_val_mapped = _map(y_val)
        y_test_mapped = _map(y_test)
        class_names_sorted = [str(x) for x in uniq_sorted]

        _emit(socketio, "ensemble_log", {"message": f"🔢 Label mapping: {lbl_map} — classes: {class_names_sorted}"})

        # base learners
        mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=200, random_state=random_state)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
        xgb = None
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0, random_state=random_state)
        except Exception:
            _emit(socketio, "ensemble_log", {"message": "⚠️ xgboost not available — skipping XGBoost model."})

        base_models = [("mlp", mlp), ("rf", rf)]
        if xgb is not None:
            base_models.append(("xgb", xgb))

        val_scores = {}
        trained_models = {}
        for idx, (name, clf) in enumerate(base_models, start=1):
            _emit(socketio, "ensemble_log", {"message": f"⚙️ Training base model {name} ({idx}/{len(base_models)})"})
            try:
                clf.fit(X_train_p, y_train_mapped)
                preds_val = clf.predict(X_val_p)
                acc = float(accuracy_score(y_val_mapped, preds_val)) if y_val_mapped.size else 0.0
                f1 = float(f1_score(y_val_mapped, preds_val, average="macro", zero_division=0)) if y_val_mapped.size else 0.0
                val_scores[name] = {"acc": acc, "f1": f1}
                trained_models[name] = clf
                _emit(socketio, "ensemble_log", {"message": f"   • {name} — val_acc: {acc:.4f}, val_f1: {f1:.4f}"})
            except Exception as ex:
                _emit(socketio, "ensemble_log", {"message": f"❌ Gagal train {name}: {ex}"})
                trained_models[name] = None

        names_present = [n for n,_ in base_models if trained_models.get(n) is not None]
        weights = np.array([val_scores.get(n, {}).get("f1", 0.0) for n in names_present], dtype=np.float32)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = (weights / weights.sum()).tolist()
        _emit(socketio, "ensemble_log", {"message": f"🎯 Voting weights (from val F1): {dict(zip(names_present, weights))}"})

        vc_estimators = [(n, trained_models[n]) for n in names_present]
        voting = VotingClassifier(estimators=vc_estimators, voting="soft", weights=weights, n_jobs=-1)
        try:
            voting.fit(X_train_p, y_train_mapped)
        except Exception:
            voting = VotingClassifier(estimators=vc_estimators, voting="soft", n_jobs=-1)
            voting.fit(X_train_p, y_train_mapped)

        estimators_for_stack = [(n, trained_models[n]) for n in names_present]
        meta_learner = LogisticRegression(max_iter=200)
        stacking = StackingClassifier(estimators=estimators_for_stack, final_estimator=meta_learner, n_jobs=-1, passthrough=False)
        stacking.fit(X_train_p, y_train_mapped)

        # Save models to models/ensemble
        for name, mdl in trained_models.items():
            if mdl is None: continue
            try:
                joblib.dump(mdl, os.path.join(models_ensemble_dir, f"{name}.pkl"))
            except Exception:
                pass
        try:
            joblib.dump(voting, os.path.join(models_ensemble_dir, "voting.pkl"))
            joblib.dump(stacking, os.path.join(models_ensemble_dir, "stacking.pkl"))
            if pca is not None:
                joblib.dump(pca, os.path.join(models_ensemble_dir, "pca.pkl"))
        except Exception:
            pass

        # Evaluate
        _emit(socketio, "ensemble_log", {"message": f"🧪 Evaluasi ensemble pada {test_for_eval} set ({X_test_p.shape[0]} samples)"})
        results_report = {}
        def eval_and_report(model_obj, name):
            try:
                ypred = model_obj.predict(X_test_p)
                acc = accuracy_score(y_test_mapped, ypred) if y_test_mapped.size else 0.0
                f1m = f1_score(y_test_mapped, ypred, average="macro", zero_division=0) if y_test_mapped.size else 0.0
                cr = classification_report(y_test_mapped, ypred, zero_division=0, output_dict=True) if y_test_mapped.size else {}
                cm = confusion_matrix(y_test_mapped, ypred).tolist() if y_test_mapped.size else []
                results_report[name] = {"accuracy": float(acc), "f1_macro": float(f1m), "confusion_matrix": cm, "report": cr}
                _emit(socketio, "ensemble_log", {"message": f"   • {name} — Acc: {acc:.4f}, F1: {f1m:.4f}"})
            except Exception as e:
                results_report[name] = {"error": str(e)}
                _emit(socketio, "ensemble_log", {"message": f"❌ Eval error {name}: {e}"})

        for n in names_present:
            eval_and_report(trained_models[n], n)
        eval_and_report(voting, "voting")
        eval_and_report(stacking, "stacking")

        # save results JSON under results/ensemble/ensemble_report.json
        out_json = os.path.join(results_ensemble_dir, "ensemble_report.json")
        with open(out_json, "w") as f:
            json.dump({
                "meta": {"processed_dir": processed_dir, "models_cnn_dir": models_cnn_dir, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                "results": results_report
            }, f, indent=2)

        # confusion plot
        try:
            cm = np.array(results_report.get("stacking", {}).get("confusion_matrix", []))
            if cm.size:
                plt.figure(figsize=(6,6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title("Confusion matrix (stacking)")
                plt.colorbar()
                tick_marks = np.arange(len(class_names_sorted))
                plt.xticks(tick_marks, class_names_sorted, rotation=45)
                plt.yticks(tick_marks, class_names_sorted)
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                plt.tight_layout()
                cm_path = os.path.join(results_ensemble_dir, "confusion_stacking.png")
                plt.savefig(cm_path)
                plt.close()
                _emit(socketio, "ensemble_log", {"message": f"📊 Confusion matrix saved: {cm_path}"})
        except Exception as e:
            _emit(socketio, "ensemble_log", {"message": f"⚠️ Could not plot confusion matrix: {e}"})

        _emit(socketio, "ensemble_done", {"message": "✅ Ensemble selesai", "report_path": out_json, "confusion_path": cm_path if 'cm_path' in locals() else None})
        return {"status":"ok", "report": out_json}

    except Exception as e:
        tb = traceback.format_exc()
        _emit(socketio, "ensemble_error", {"message": str(e), "trace": tb})
        raise

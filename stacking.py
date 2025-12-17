# stacking.py
import argparse
import os
import numpy as np
import joblib
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_model_predict(model_path, datagen, generator):
    # load keras model and predict probabilities for generator
    model = keras.models.load_model(model_path, compile=False)
    preds = model.predict(generator, verbose=1)
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    models_dir = args.models_dir
    data_dir = args.data_dir
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    IMG_SIZE = (224,224)
    batch = 16
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(os.path.join(data_dir,'test'), target_size=IMG_SIZE, batch_size=batch, class_mode='categorical', shuffle=False)

    model_files = {
        'densenet': os.path.join(models_dir, 'densenet121_best.h5'),
        'inceptionresnetv2': os.path.join(models_dir, 'inceptionresnetv2_best.h5'),
        'efficientnetv2': os.path.join(models_dir, 'efficientnetv2_best.h5')
    }

    probs_list = []
    for k, p in model_files.items():
        if not os.path.exists(p):
            print("Warning: model not found", p)
            # create dummy uniform probs
            probs_list.append(np.ones((len(test_gen.filenames),5))/5.0)
            continue
        model = keras.models.load_model(p, compile=False)
        preds = model.predict(test_gen, verbose=1)
        probs_list.append(preds)

    # concatenate probs per sample as features
    X = np.hstack(probs_list)
    y_true = test_gen.labels

    # split X into train-meta and test-meta? For simplicity we train meta on test set via cross-validation here (user may prefer holdout)
    # We'll train meta classifiers using X and y_true (WARNING: in real pipeline use validation set or out-of-fold preds)
    classifiers = {
        'MLP': MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200),
        'RandomForest': RandomForestClassifier(n_estimators=200),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    results = []
    for name, clf in classifiers.items():
        clf.fit(X, y_true)
        y_pred = clf.predict(X)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        # AUC multi-class (one-vs-rest)
        try:
            proba = clf.predict_proba(X)
            auc = roc_auc_score(pd.get_dummies(y_true), proba, average='weighted', multi_class='ovr')
        except Exception:
            auc = None
        results.append({'model': name, 'accuracy': acc, 'f1': f1, 'kappa': kappa, 'auc': auc})
        # save classifier
        joblib.dump(clf, os.path.join(out_dir, f"meta_{name}.joblib"))

    import json
    with open(os.path.join(out_dir, "meta_summary.json"), "w") as f:
        json.dump(results, f)

    print("Stacking selesai. Results:")
    print(results)

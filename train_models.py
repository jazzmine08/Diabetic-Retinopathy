# train_models.py
import argparse
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers
import numpy as np

def build_model(name, input_shape=(224,224,3), num_classes=5):
    if name == 'densenet121':
        base = keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    elif name == 'inceptionresnetv2':
        base = keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    elif name == 'efficientnetv2':
        base = keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    else:
        raise ValueError("Unknown model")
    x = base.output
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base.input, outputs=out)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_dir = args.data_dir
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    IMG_SIZE = (224,224)
    BATCH = args.batch
    EPOCHS = args.epochs
    num_classes = 5

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.15,
                                       horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(os.path.join(data_dir, 'train'), target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(os.path.join(data_dir, 'val'), target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

    models_to_train = ['densenet121','inceptionresnetv2','efficientnetv2']
    summary = []
    for name in models_to_train:
        print("Training", name)
        model = build_model(name, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3), num_classes=num_classes)
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        ckpt = os.path.join(models_dir, f"{name}_best.h5")
        callbacks = [
            ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        ]
        start = time.time()
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
        elapsed = time.time() - start
        # Evaluate on validation (or test after)
        val_loss, val_acc = model.evaluate(val_gen)
        summary.append({'model': name, 'val_acc': float(val_acc), 'val_loss': float(val_loss), 'train_time_s': elapsed})
        # save final history
        import json
        with open(os.path.join(models_dir, f"{name}_history.json"), "w") as f:
            json.dump(history.history, f)
    # save summary
    import json
    with open(os.path.join(models_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f)
    print("Training semua model selesai. Summary tersimpan.")

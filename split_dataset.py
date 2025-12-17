import os
import shutil
import numpy as np

def split_dataset(processed_dir, output_dir):
    classes = ["0", "1", "2", "3", "4"]

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # Buat folder output
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Proses setiap label
    for cls in classes:
        class_path = os.path.join(processed_dir, cls)
        images = os.listdir(class_path)

        np.random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        for f in train_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(output_dir, "train", cls)
            )

        for f in val_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(output_dir, "val", cls)
            )

        for f in test_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(output_dir, "test", cls)
            )

    return True

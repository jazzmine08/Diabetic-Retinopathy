import os

# Lokasi folder dataset tempat gambar berada
DATASET_DIR = r"D:\Improving Diabetic Retinopathy Grading Accuracy\dataset"

# Kata atau pola yang ingin dihapus dari nama file
REMOVE_KEYWORDS = ["Salinan ", "Copy of ", "copy of ", "salinan "]

count = 0
for filename in os.listdir(DATASET_DIR):
    old_path = os.path.join(DATASET_DIR, filename)

    if not os.path.isfile(old_path):
        continue

    name, ext = os.path.splitext(filename)
    new_name = name

    # Hapus kata kunci dari nama file
    for key in REMOVE_KEYWORDS:
        if new_name.startswith(key):
            new_name = new_name.replace(key, "")
    
    new_filename = new_name.strip() + ext
    new_path = os.path.join(DATASET_DIR, new_filename)

    if new_path != old_path:
        os.rename(old_path, new_path)
        count += 1
        print(f"Renamed: {filename} → {new_filename}")

print(f"\n✅ Selesai! Total file diubah namanya: {count}")

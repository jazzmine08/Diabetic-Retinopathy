import os
import shutil
import pandas as pd
from difflib import get_close_matches

# === KONFIGURASI ===
SOURCE_DIR = r"D:\Improving Diabetic Retinopathy Grading Accuracy\dataset"
CSV_FILE = os.path.join(SOURCE_DIR, "label_grade.csv")
OUTPUT_DIR = os.path.join("dataset", "raw")

# === BACA CSV LABEL ===
df = pd.read_csv(CSV_FILE)
print(f"Total data dalam CSV: {len(df)}")

# === AMBIL SEMUA FILE GAMBAR ===
image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_basenames = [os.path.splitext(f)[0].lower() for f in image_files]

# === BUAT FOLDER TUJUAN ===
for label in sorted(df['label'].unique()):
    os.makedirs(os.path.join(OUTPUT_DIR, str(label)), exist_ok=True)

# === FUNGSI PEMBANTU ===
def find_best_match(filename):
    name = os.path.splitext(filename)[0].lower()
    match = get_close_matches(name, image_basenames, n=1, cutoff=0.6)
    if match:
        idx = image_basenames.index(match[0])
        return image_files[idx]
    return None

# === PROSES PEMINDAHAN FILE ===
not_found = []
for _, row in df.iterrows():
    filename = str(row['filename'])
    label = str(row['label'])
    matched = find_best_match(filename)
    
    if matched:
        src = os.path.join(SOURCE_DIR, matched)
        dst_dir = os.path.join(OUTPUT_DIR, label)
        dst = os.path.join(dst_dir, matched)
        shutil.copy(src, dst)
    else:
        not_found.append(filename)

print(f"\n✅ Dataset berhasil disusun ke dalam folder: {OUTPUT_DIR}")
print(f"Total gambar cocok: {len(df) - len(not_found)} / {len(df)}")
if not_found:
    print("\n⚠️ Tidak ditemukan kecocokan untuk:")
    for n in not_found[:10]:  # tampilkan hanya 10 pertama
        print(" -", n)

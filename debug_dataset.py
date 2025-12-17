import os

base = r"D:\Improving Diabetic Retinopathy Grading Accuracy\dataset\processed_final"

for split in ["train", "val", "test"]:
    path = os.path.join(base, split)
    print(f"\n📁 {split.upper()} — {path}")

    if not os.path.exists(path):
        print("❌ Folder tidak ada!")
        continue

    for cls in sorted(os.listdir(path)):
        cpath = os.path.join(path, cls)
        if os.path.isdir(cpath):
            files = [f for f in os.listdir(cpath) if f.lower().endswith((".jpg",".png",".jpeg"))]
            print(f"  Kelas {cls}: {len(files)} gambar")

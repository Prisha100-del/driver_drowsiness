import os
import random
import shutil

BASE_DIR = "dataset_yawn_cropped"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

CLASSES = ["yawn", "no_yawn"]
SPLIT_COUNT = 100  # number of images per class for validation

for cls in CLASSES:
    train_class_path = os.path.join(TRAIN_DIR, cls)
    val_class_path = os.path.join(VAL_DIR, cls)

    os.makedirs(val_class_path, exist_ok=True)

    images = os.listdir(train_class_path)
    random.shuffle(images)

    val_images = images[:SPLIT_COUNT]

    for img in val_images:
        src = os.path.join(train_class_path, img)
        dst = os.path.join(val_class_path, img)
        shutil.move(src, dst)

    print(f"{cls}: moved {len(val_images)} images to validation")

print("Dataset split complete!")

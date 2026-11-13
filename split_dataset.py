import os
import random
import shutil

# Dataset path
dataset_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset"

classes = ["Good", "Average", "Poor"]

train_ratio = 0.70
val_ratio = 0.15

random.seed(42)

for cls in classes:
    source = os.path.join(dataset_path, cls)

    images = os.listdir(source)
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = images[:train_end]
    validation = images[train_end:val_end]
    test = images[val_end:]

    for img in train:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(dataset_path, "train", cls, img)
        )

    for img in validation:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(dataset_path, "validation", cls, img)
        )

    for img in test:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(dataset_path, "test", cls, img)
        )

print("Dataset split completed successfully!")
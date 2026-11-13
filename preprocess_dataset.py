import os
import cv2

# Dataset paths
input_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset"
output_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset"

folders = ["train", "validation", "test"]
classes = ["Good", "Average", "Poor"]

for folder in folders:
    for cls in classes:

        input_folder = os.path.join(input_path, folder, cls)
        output_folder = os.path.join(output_path, folder, cls)

        os.makedirs(output_folder, exist_ok=True)

        for image_name in os.listdir(input_folder):

            # Skip non-image files
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(input_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                print("Cannot read:", image_path)
                continue

            # Resize
            image = cv2.resize(image, (224, 224))

            # Slight contrast improvement
            image = cv2.convertScaleAbs(
                image,
                alpha=1.1,
                beta=5
            )

            # Save
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, image)

print("✅ All images have been preprocessed successfully!")
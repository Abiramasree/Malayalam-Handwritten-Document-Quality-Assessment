import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load trained model
model = load_model("document_quality_model.keras")

# Class names
class_names = ["Average", "Good", "Poor"]

# Image path
image_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\Good\good (8).png"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print(image_path)
    exit()

# Resize
image = cv2.resize(image, (224, 224))

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to float
image = np.array(image, dtype=np.float32)

# ResNet50 preprocessing
image = preprocess_input(image)

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)

# Predicted class
predicted_class = np.argmax(prediction)

# Confidence
confidence = np.max(prediction) * 100

# Display result
print("\nPrediction Result")
print("-----------------")
print("Predicted Class :", class_names[predicted_class])
print(f"Confidence      : {confidence:.2f}%")
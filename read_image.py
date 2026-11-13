import cv2
import matplotlib.pyplot as plt

# Image path
image_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\train\Good\good (8).png"

# Read image
image = cv2.imread(image_path)

# Convert BGR to RGB (for displaying the original image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Display Original and Grayscale images


plt.figure(figsize=(25,5))

# Original
plt.subplot(1,5,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

# Grayscale
plt.subplot(1,5,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# Gaussian Blur
plt.subplot(1,5,3)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

# Threshold
plt.subplot(1,5,4)
plt.imshow(binary, cmap="gray")
plt.title("Threshold")
plt.axis("off")

# Morphological Output
plt.subplot(1,5,5)
plt.imshow(morph, cmap="gray")
plt.title("Morphology")
plt.axis("off")

plt.show()
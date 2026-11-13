import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===================================
# Dataset Paths
# ===================================

train_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\train"
validation_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\validation"
test_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\test"

# ===================================
# Load Dataset
# ===================================

train_dataset = image_dataset_from_directory(
    train_path,
    image_size=(224,224),
    batch_size=16,
    label_mode="categorical"
)

validation_dataset = image_dataset_from_directory(
    validation_path,
    image_size=(224,224),
    batch_size=16,
    label_mode="categorical"
)

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=(224,224),
    batch_size=16,
    label_mode="categorical"
)

# ===================================
# Data Augmentation
# ===================================

data_augmentation = tf.keras.Sequential([
    RandomFlip("vertical"),
    RandomRotation(0.05),
    RandomZoom(0.05),
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(preprocess_input(x), training=True), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

# ===================================
# Load ResNet50
# ===================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze most layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ===================================
# Build Model
# ===================================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(3, activation="softmax")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ===================================
# Compile
# ===================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===================================
# Callbacks
# ===================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2
)

# ===================================
# Train
# ===================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ===================================
# Evaluate
# ===================================

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

model.save("document_quality_model.keras")

print("\nModel saved successfully!")
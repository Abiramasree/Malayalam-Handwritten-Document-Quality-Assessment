import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==========================================
# Dataset Paths
# ==========================================

train_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\train"
validation_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\validation"
test_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\test"

# ==========================================
# Load Dataset
# ==========================================

train_dataset = image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    shuffle=True
)

validation_dataset = image_dataset_from_directory(
    validation_path,
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    shuffle=False
)

# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([
    RandomRotation(0.03),
    RandomZoom(0.08),
    RandomContrast(0.10)
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ==========================================
# Performance Optimization
# ==========================================

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# ==========================================
# Load EfficientNetB0
# ==========================================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model initially
base_model.trainable = False

# ==========================================
# Build Model
# ==========================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

output = Dense(
    3,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ==========================================
# Compile
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# Callbacks
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_efficientnetB0.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)

# ==========================================
# Training
# ==========================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# ==========================================
# Evaluation
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("\n====================================")
print("EfficientNetB0 Results")
print("====================================")

print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ==========================================
# Save Final Model
# ==========================================

model.save(
    "efficientnetB0_document_quality_model.keras"
)

print("\nModel saved successfully!")
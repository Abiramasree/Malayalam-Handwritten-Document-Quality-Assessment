import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# DATASET PATHS
# ============================================================

train_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\train"

validation_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\validation"

test_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\test"


# ============================================================
# LOAD DATASET
# ============================================================

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


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    RandomRotation(0.03),
    RandomZoom(0.08),
    RandomContrast(0.10)
])


train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# PERFORMANCE
# ============================================================

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# LOAD RESNET50
# ============================================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initially freeze the complete pretrained model
base_model.trainable = False


# ============================================================
# BUILD CLASSIFICATION HEAD
# ============================================================

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


# ============================================================
# STAGE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n========================================")
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("========================================\n")


history1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


# ============================================================
# STAGE 2 - FINE-TUNING
# ============================================================

print("\n========================================")
print("STAGE 2: RESNET50 FINE-TUNING")
print("========================================\n")


# Unfreeze the ResNet50 model
base_model.trainable = True


# Freeze earlier layers
# Fine-tune only the last 30 layers

for layer in base_model.layers[:-30]:
    layer.trainable = False


# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


# Recompile with a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    "best_resnet50_finetuned.keras",
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


# ============================================================
# FINE-TUNE
# ============================================================

history2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# ============================================================
# FINAL EVALUATION
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


print("\n========================================")
print("RESNET50 FINE-TUNING RESULTS")
print("========================================")

print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "resnet50_finetuned_document_quality_model.keras"
)

print("\nModel saved successfully!")
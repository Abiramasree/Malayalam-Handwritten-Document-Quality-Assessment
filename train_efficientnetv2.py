import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    RandomRotation,
    RandomZoom,
    RandomContrast
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. DATASET PATHS
# ============================================================

train_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\train"

validation_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\validation"

test_path = r"C:\Users\Abiramasree K\Desktop\Malayalam_Dataset\preprocess_dataset\test"


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3


# ============================================================
# 3. LOAD TRAIN DATASET
# ============================================================

train_dataset = image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)


# ============================================================
# 4. LOAD VALIDATION DATASET
# ============================================================

validation_dataset = image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# 5. LOAD TEST DATASET
# ============================================================

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# 6. DISPLAY CLASS NAMES
# ============================================================

print("\n======================================")
print("CLASS NAMES")
print("======================================")

print(train_dataset.class_names)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    RandomRotation(0.03),
    RandomZoom(0.08),
    RandomContrast(0.10)
])


# ============================================================
# 8. APPLY DATA AUGMENTATION
# ============================================================

train_dataset = train_dataset.map(
    lambda x, y: (
        data_augmentation(x, training=True),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 9. PERFORMANCE OPTIMIZATION
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 10. LOAD EfficientNetV2B0
# ============================================================

print("\n======================================")
print("Loading EfficientNetV2B0...")
print("======================================")


base_model = EfficientNetV2B0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# 11. FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# 12. BUILD CLASSIFICATION HEAD
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


# ============================================================
# 13. CREATE FINAL MODEL
# ============================================================

model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 14. DISPLAY MODEL
# ============================================================

print("\n======================================")
print("MODEL SUMMARY")
print("======================================")

model.summary()


# ============================================================
# 15. COMPILE MODEL - STAGE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 16. CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    "best_efficientnetv2B0.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 17. INITIAL TRAINING
# ============================================================

print("\n======================================")
print("STAGE 1: INITIAL TRAINING")
print("======================================")


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# ============================================================
# 18. FINE-TUNING
# ============================================================

print("\n======================================")
print("STAGE 2: FINE-TUNING")
print("======================================")


# Unfreeze the base model
base_model.trainable = True


# Freeze most layers
for layer in base_model.layers[:-40]:
    layer.trainable = False


# ============================================================
# 19. COMPILE AGAIN WITH LOW LEARNING RATE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 20. FINE-TUNING CALLBACKS
# ============================================================

early_stop_finetune = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


checkpoint_finetune = ModelCheckpoint(
    "best_efficientnetv2B0_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


reduce_lr_finetune = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 21. FINE-TUNE MODEL
# ============================================================

history_finetune = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[
        early_stop_finetune,
        checkpoint_finetune,
        reduce_lr_finetune
    ]
)


# ============================================================
# 22. EVALUATE ON TEST DATA
# ============================================================

print("\n======================================")
print("FINAL EVALUATION")
print("======================================")


test_loss, test_accuracy = model.evaluate(
    test_dataset
)


# ============================================================
# 23. PRINT RESULTS
# ============================================================

print("\n======================================")
print("EfficientNetV2B0 RESULTS")
print("======================================")

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss     : {test_loss:.4f}"
)


# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

model.save(
    "efficientnetv2B0_document_quality_model.keras"
)


# ============================================================
# 25. FINISHED
# ============================================================

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")

print(
    "efficientnetv2B0_document_quality_model.keras"
)
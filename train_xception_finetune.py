import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

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
# 2. PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("\n======================================")
print("LOADING DATASETS")
print("======================================")

train_dataset = image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

validation_dataset = image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# 4. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\n======================================")
print("CLASSES")
print("======================================")

print(class_names)


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    RandomRotation(0.03),
    RandomZoom(0.08),
    RandomContrast(0.10)
])


# ============================================================
# 6. AUGMENT TRAINING DATA
# ============================================================

train_dataset = train_dataset.map(
    lambda x, y: (
        data_augmentation(x, training=True),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 7. XCEPTION PREPROCESSING
# ============================================================

train_dataset = train_dataset.map(
    lambda x, y: (
        preprocess_input(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda x, y: (
        preprocess_input(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    lambda x, y: (
        preprocess_input(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 8. PREFETCH
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
# 9. LOAD XCEPTION
# ============================================================

print("\n======================================")
print("LOADING XCEPTION")
print("======================================")

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ============================================================
# 10. FREEZE XCEPTION
# ============================================================

base_model.trainable = False


# ============================================================
# 11. BUILD CLASSIFICATION HEAD
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


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 12. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 13. STAGE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n======================================")
print("STAGE 1: INITIAL TRAINING")
print("======================================")


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 14. CALLBACKS - STAGE 1
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_xception_initial.keras",
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
# 15. TRAIN STAGE 1
# ============================================================

history1 = model.fit(
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
# 16. STAGE 2 - FINE-TUNING
# ============================================================

print("\n======================================")
print("STAGE 2: XCEPTION FINE-TUNING")
print("======================================")


# Unfreeze Xception

base_model.trainable = True


# Freeze early layers
# Train only the last 30 layers

for layer in base_model.layers[:-30]:
    layer.trainable = False


# ============================================================
# 17. COMPILE FOR FINE-TUNING
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 18. CALLBACKS - FINE-TUNING
# ============================================================

early_stop_finetune = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint_finetune = ModelCheckpoint(
    "best_xception_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

reduce_lr_finetune = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)


# ============================================================
# 19. TRAIN FINE-TUNING
# ============================================================

history2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[
        early_stop_finetune,
        checkpoint_finetune,
        reduce_lr_finetune
    ]
)


# ============================================================
# 20. FINAL EVALUATION
# ============================================================

print("\n======================================")
print("FINAL EVALUATION")
print("======================================")


test_loss, test_accuracy = model.evaluate(
    test_dataset
)


# ============================================================
# 21. RESULTS
# ============================================================

print("\n======================================")
print("XCEPTION FINE-TUNING RESULTS")
print("======================================")

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss     : {test_loss:.4f}"
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

model.save(
    "xception_finetuned_document_quality_model.keras"
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")

print(
    "xception_finetuned_document_quality_model.keras"
)

print("\nClasses:", class_names)

print(
    f"Final Test Accuracy: {test_accuracy * 100:.2f}%"
)
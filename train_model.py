# =================== train_model.py ===================
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Paths
train_dir = "C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/emotion detection/train"
test_dir = "C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/emotion detection/test"

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode="categorical")

# Class Weights
train_classes = np.array(train_generator.classes)
class_labels = np.unique(train_classes)
class_weights = compute_class_weight('balanced', classes=class_labels, y=train_classes)
class_weights_dict = dict(enumerate(np.clip(class_weights, 0.5, 2.0)))

# Model
model = Sequential([
    Input(shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Training
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_reduce]
)

# Plotting
def plot_metrics(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    plt.show()

plot_metrics(history)

# Save history for later evaluation
np.save("C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/project_code/training_history.npy", history.history)


# Save Model
model.save("C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/project_code/emotion_detection_cnn_model.h5")

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"🎯 Test Accuracy: {acc*100:.2f}% | Test Loss: {loss:.4f}")

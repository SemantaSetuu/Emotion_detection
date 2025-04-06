#evaluate_model.py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import random

# Load the Trained Model
model_path = r"C:\Users\seman\Desktop\clg\2nd_sem\generative_AI\project_code\CNN_emotion_detection_model.h5"
model = tf.keras.models.load_model(model_path)

#Load Test Data
test_dir = "C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/emotion detection/test"
img_size = (48, 48)
batch_size = 128

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

#Evaluate Accuracy & Loss
loss, acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}\n")

#Generate Predictions
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

#confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")


plt.show()

#Classification Report (includes support)
report = classification_report(y_true, y_pred, target_names=class_labels)
print("\nClassification Report:\n")
print(report)

#Accuracy & Loss Plot
history_path = r"C:\Users\seman\Desktop\clg\2nd_sem\generative_AI\project_code\training_history.npy"
if os.path.exists(history_path):
    history = np.load(history_path, allow_pickle=True).item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    plt.show()

#Random 6–7 Predictions (Visual Check)
print("\nSample Predictions:\n")
fig, axes = plt.subplots(2, 4, figsize=(12,6))
fig.suptitle("Random Test Samples with Predicted vs True Labels")

files = test_generator.filenames
sample_indices = random.sample(range(len(files)), 7)

for i, idx in enumerate(sample_indices):
    img_path = os.path.join(test_dir, files[idx])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0
    input_img = np.expand_dims(img_norm, axis=(0, -1))
    pred_probs = model.predict(input_img)[0]
    pred_idx = np.argmax(pred_probs)
    pred_label = class_labels[pred_idx]
    confidence = pred_probs[pred_idx] * 100
    true_label = class_labels[y_true[idx]]

    ax = axes[i//4, i%4]
    ax.imshow(img_resized, cmap='gray')
    ax.set_title(f"Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}")
    ax.axis('off')

plt.tight_layout()
plt.show()

#Bonus: Show Misclassified Samples Only
print("\n Misclassified Samples:\n")
wrong_indices = np.where(y_true != y_pred)[0]
if len(wrong_indices) == 0:
    print("No misclassified samples!")
else:
    sample_wrong = random.sample(list(wrong_indices), min(6, len(wrong_indices)))
    fig, axes = plt.subplots(1, len(sample_wrong), figsize=(15,4))
    fig.suptitle("Misclassified Examples")
    for i, idx in enumerate(sample_wrong):
        img_path = os.path.join(test_dir, files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, img_size)
        ax = axes[i]
        ax.imshow(img_resized, cmap='gray')
        pred_probs = model.predict(np.expand_dims(img_resized / 255.0, axis=(0, -1)))[0]
        pred_idx = np.argmax(pred_probs)
        confidence = pred_probs[pred_idx] * 100
        ax.set_title(f"Pred: {class_labels[pred_idx]} ({confidence:.1f}%)\nTrue: {class_labels[y_true[idx]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
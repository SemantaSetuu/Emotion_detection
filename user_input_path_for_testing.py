# =================== user_input_path_for_testing.py ===================
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ========== 1. Load the trained model ==========
model = tf.keras.models.load_model("C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/project_code/emotion_detection_cnn_model.h5")

# ========== 2. Define emotion classes (order must match training) ==========
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# ========== 3. Take user input ==========
image_path = input("🖼️ Enter full path of the image: ").strip()

# ========== 4. Load and preprocess the image ==========
try:
    # Read in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or unsupported format.")

    # Resize to 48x48
    resized_img = cv2.resize(image, (48, 48))

    # Normalize and reshape
    input_img = resized_img.astype("float32") / 255.0
    input_img = np.expand_dims(input_img, axis=0)  # Batch dimension
    input_img = np.expand_dims(input_img, axis=-1) # Channel dimension (grayscale)

    # Predict
    prediction = model.predict(input_img)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Show result
    print(f"🔍 Predicted Emotion: {predicted_class.upper()} ({confidence:.1f}%)")

    # Optional: show image with prediction
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"Predicted: {predicted_class} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()

except Exception as e:
    print("Error:", e)
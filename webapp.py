import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ========== Load the Trained Model ==========
model_path = r"C:/Users/seman/Desktop/fer2013_cnn_local.h5"
model = tf.keras.models.load_model(model_path)

# ========== Define Emotion Labels ==========
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# ========== Title ==========
st.title("😊 Real-time Emotion Detection Web App")
st.write("Upload an image or take a picture with your webcam to predict the emotion.")

# ========== Upload or Capture ==========
option = st.radio("Choose input method:", ('Upload Image', 'Use Webcam'))

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype("float32") / 255.0
    input_img = np.expand_dims(normalized, axis=(0, -1))  # (1, 48, 48, 1)
    return input_img, resized

# ========== Upload ==========
if option == 'Upload Image':
    uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        input_img, resized_img = preprocess_image(img_np)

        prediction = model.predict(input_img)[0]
        predicted_idx = np.argmax(prediction)
        predicted_label = class_names[predicted_idx]
        confidence = prediction[predicted_idx] * 100

        st.image(img, caption=f"Predicted: {predicted_label.upper()} ({confidence:.1f}%)", use_column_width=True)

# ========== Webcam ==========
elif option == 'Use Webcam':
    picture = st.camera_input("Take a photo")
    if picture is not None:
        img = Image.open(picture).convert("RGB")
        img_np = np.array(img)
        input_img, resized_img = preprocess_image(img_np)

        prediction = model.predict(input_img)[0]
        predicted_idx = np.argmax(prediction)
        predicted_label = class_names[predicted_idx]
        confidence = prediction[predicted_idx] * 100

        st.image(img, caption=f"Predicted: {predicted_label.upper()} ({confidence:.1f}%)", use_column_width=True)

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import pandas as pd
import altair as alt
import datetime
import os

# Load trained model
model = tf.keras.models.load_model("C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/project_code/emotion_detection_cnn_model.h5")

# Define emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

st.set_page_config(page_title="Emotion Detection Web App", layout="centered")
st.title("😊 Real-Time & Image-based Emotion Detection")
st.write("Upload an image, use your webcam, or capture a live photo to detect emotions.")

# Upload image section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        resized_img = cv2.resize(image, (48, 48))
        input_img = resized_img.astype("float32") / 255.0
        input_img = np.expand_dims(input_img, axis=(0, -1))

        prediction = model.predict(input_img)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = emotion_labels[predicted_index]
        confidence = prediction[predicted_index] * 100

        st.image(cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        st.markdown(f"### 🎯 Predicted Emotion: **{predicted_label.upper()} ({confidence:.1f}%)**")

        # Plot bar chart of all predictions
        chart_df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Confidence (%)": prediction * 100
        })
        bar_chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Emotion", sort="-y"),
            y="Confidence (%)",
            color=alt.Color("Emotion", scale=alt.Scale(scheme='tableau20'))
        )
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.error("⚠️ Unable to read image. Please upload a valid image file.")

st.markdown("---")
st.subheader("📸 Real-Time Webcam Emotion Detection")

if st.button("▶️ Start Real-Time Camera Prediction"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    st.info("Press 'Q' on the camera window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=(0, -1))

            prediction = model.predict(roi_input)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = emotion_labels[predicted_index]
            confidence = prediction[predicted_index] * 100

            label_text = f"{predicted_label} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Detection - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

st.markdown("---")
st.subheader("📷 Quick Capture and Predict")

if st.button("📸 Capture and Predict from Camera"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Failed to capture image from camera.")
    else:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected. Try again.")
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=(0, -1))

                prediction = model.predict(roi_input)[0]
                predicted_index = np.argmax(prediction)
                predicted_label = emotion_labels[predicted_index]
                confidence = prediction[predicted_index] * 100

                st.image(frame, channels="BGR", caption="Quick Captured Frame")
                st.markdown(f"### 🎯 Predicted Emotion: **{predicted_label.upper()} ({confidence:.1f}%)**")

                # Bar chart of prediction confidence
                chart_df = pd.DataFrame({
                    "Emotion": emotion_labels,
                    "Confidence (%)": prediction * 100
                })
                bar_chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Emotion", sort="-y"),
                    y="Confidence (%)",
                    color=alt.Color("Emotion", scale=alt.Scale(scheme='tableau20'))
                )
                st.altair_chart(bar_chart, use_container_width=True)
                break

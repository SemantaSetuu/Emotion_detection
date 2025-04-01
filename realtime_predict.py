# =================== realtime_predict.py ===================
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model_path = "C:/Users/seman/Desktop/clg/2nd_sem/generative_AI/emotion detection/project_code/emotion_detection_cnn_model.h5"
model = tf.keras.models.load_model(model_path)

# Define emotion classes (same order as training)
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to exit real-time detection.")

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

        prediction = model.predict(roi_input)
        predicted_index = np.argmax(prediction)
        predicted_label = emotion_labels[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        # Draw rectangle & label with confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_text = f"{predicted_label} ({confidence:.1f}%)"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








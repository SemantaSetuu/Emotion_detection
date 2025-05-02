#  Real-Time Emotion Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to detect human emotions from facial expressions in grayscale images. The model is trained on categorized emotion images and deployed with a Streamlit-based web application for real-time and image-based emotion prediction.

---

##  Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **Streamlit**
- **Altair (for charts)**

---
##  Dataset Description: FER-2013 Facial Expression Dataset
The FER-2013 dataset is a publicly available facial expression dataset originally published for the ICML 2013 Challenges in Representation Learning. It is commonly used for training emotion recognition models using deep learning.
Structure:
The dataset contains grayscale facial images with a resolution of 48x48 pixels categorized into 7 emotion classes:

Angry,
Disgust,
Fear,
Happy,
Neutral,
Sad,
Surprise

The dataset is organized into two folders:
train/: Training images for each emotion class.
test/: Testing images for each emotion class.

Each class folder contains image files (.jpg) of faces showing that emotion.
Image Format: 48x48 grayscale
Classes: 7
Labels: Stored as folder names
Purpose: Emotion detection via facial expressions
🔗 Source:
Dataset link: FER-2013 on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)

---

##  Folder Structure

project_code/
├── emotion detection datasets/ # Folder containing training and test datasets
│ ├── train/
│ └── test/
├── train_model.py # CNN training script
├── evaluation_model.py # Model evaluation, confusion matrix, and test output
├── training_history.npy # Saved training metrics
├── CNN_emotion_detection_model.h5 # Trained model
├── webapp.py # Streamlit app for image/webcam-based prediction
├── requirements.txt # Python dependencies

---

##  How to Run

### 1. Install required libraries
pip install -r requirements.txt

2. Train the model (optional, already trained model is included)
run:
python train_model.py

3. Evaluate the model
run:
python evaluation_model.py

4. Launch the Streamlit Web Application
run below command on terminal:
streamlit run webapp.py

You will get the following functionalities:
-Upload a face image to detect emotion
-Use webcam for real-time emotion prediction
-Take a quick snapshot and get prediction instantly


Features
CNN trained on 48x48 grayscale face images

Supports 7 emotion classes: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised

Real-time webcam emotion prediction

Upload-based prediction with confidence scores

Accuracy and loss plots

Confusion matrix and classification report

Visualization of wrong predictions



##  Requirements
Python 3.10
TensorFlow
Streamlit
OpenCV
NumPy
Seaborn
Matplotlib
scikit-learn
Altair

Install all from:
pip install -r requirements.txt


##  Example Outputs:
Test Accuracy printed after evaluation
Confusion Matrix plotted using Seaborn
Bar chart showing prediction confidence for each emotion
Live prediction with bounding boxes using OpenCV in the Streamlit app

##  Notes
Dataset folder emotion detection should be in the same directory.

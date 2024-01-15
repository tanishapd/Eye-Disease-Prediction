import streamlit as st
import numpy as np
import cv2
#from tensorflow import keras
import os
import joblib

model = joblib.load("C:/Users/KIIT/Desktop/eye disease(streamlit)/EYE DISEASE PREDICTION.joblib")


def classify_image(image_path):
    # Preprocess the image as you did in your code
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (250, 250))
    image = image / 255.0
    image = image.reshape(1, 250, 250, 3)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return predicted_class

def main():
    st.title("EYE DISEASE PREDICTOR")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        if not os.path.exists("uploaded_images"):
            os.mkdir("uploaded_images")

        image_path = os.path.join("uploaded_images", "uploaded_image.png")
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        pred_class = ''
        if st.button('Eye disease prediction Result'):
            pred_class = classify_image(image_path)

        st.write(f'Predicted Class: {pred_class}')

if __name__ == '__main__':
    main()

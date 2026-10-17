
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

MODEL_PATH = "notebook/brain_tumor_best.h5"

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(model, image):
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    return predicted_label, confidence, predictions


st.title("Brain Tumor Detection")
st.subheader("Upload a brain MRI image to classify it.")
st.write("Supported formats: JPEG, JPG, BMP, PNG")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpeg", "jpg", "bmp", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=350)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                model = load_model()
                predicted_label, confidence, predictions = predict_image(
                    model, image
                )

            st.divider()
            st.subheader("Prediction Results")

            if predicted_label == "notumor":
                st.success("Predicted class: No Tumor")
            else:
                st.warning(f"Predicted class: {predicted_label}")

            st.metric("Confidence", f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {e}")

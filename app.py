import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

# ---------------------------
# CONFIG
# ---------------------------
MODEL_URL = "https://drive.google.com/uc?id=177UheL7E9YubKxQh74n1KokLQUFCdXcE"
MODEL_PATH = "final_model.h5"

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'other', 'pituitary']

IMG_SIZE = 224

# ---------------------------
# DOWNLOAD MODEL (if not exists)
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to predict tumor type")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ---------------------------
# PREDICTION
# ---------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess_image(image)

    prediction = model.predict(img)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction")
    st.write(f"🧾 Class: **{predicted_class}**")
    st.write(f"📊 Confidence: **{confidence*100:.2f}%**")

    # Probabilities
    st.subheader("Class Probabilities")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {prediction[i]*100:.2f}%")

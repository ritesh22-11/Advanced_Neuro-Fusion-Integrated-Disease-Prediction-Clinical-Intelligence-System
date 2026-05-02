import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

# ---------------- CONFIG ----------------
MODEL_URL = "https://drive.google.com/uc?id=1vc55Y2litNzNXgXl6LaO4iE3BuuJnu2n"
MODEL_PATH = "model.h5"

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'other', 'pituitary']
IMG_SIZE = 224

# ---------------- DOWNLOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    # 🔥 IMPORTANT FIX
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

model = load_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Brain Tumor Classifier")
st.title("🧠 Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- PREDICT ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    x = preprocess(image)
    preds = model.predict(x)[0]

    idx = np.argmax(preds)
    conf = preds[idx]

    st.success(f"Prediction: {CLASS_NAMES[idx]}")
    st.write(f"Confidence: {conf*100:.2f}%")

    st.subheader("All Probabilities")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"{c}: {preds[i]*100:.2f}%")

import os
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "best_model.keras"
FILE_ID = "1YLUblgBMrSgZZKSPGEbF0FpzKQpCW-Mt"

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'other', 'pituitary']

IMG_SIZE = (224, 224)

# ==============================
# DOWNLOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... ⏳"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# ==============================
# PREPROCESSING (VGG)
# ==============================

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32)

    # RGB → BGR
    img = img[..., ::-1]

    # VGG mean subtraction
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68

    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# GRADCAM
# ==============================

def make_gradcam(model, img_array, last_conv_layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

# ==============================
# UI
# ==============================

st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

st.title("🧠 Brain Tumor Classification System")
st.write("Upload an MRI image to predict tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# ==============================
# MAIN LOGIC
# ==============================

if uploaded_file:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(image)

        with st.spinner("Analyzing image..."):
            preds = model.predict(img_array)[0]

        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds)

        with col2:
            st.subheader("Prediction")
            st.success(f"{pred_class.upper()} ({confidence*100:.2f}%)")

            st.subheader("Confidence Scores")
            for i, cls in enumerate(CLASS_NAMES):
                st.write(f"{cls}: {preds[i]*100:.2f}%")
                st.progress(float(preds[i]))

        # ==============================
        # GradCAM Visualization
        # ==============================

        heatmap = make_gradcam(model, img_array)

        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)

        img = np.array(image.resize(IMG_SIZE))

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        st.subheader("GradCAM Visualization")
        st.image(superimposed_img.astype(np.uint8), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown("⚠️ This tool is for research purposes only. Not for clinical use.")

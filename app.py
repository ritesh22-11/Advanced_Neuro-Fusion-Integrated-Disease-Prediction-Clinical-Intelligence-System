import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        color: #1a1a2e; text-align: center;
        padding: 1rem 0 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666;
        text-align: center; margin-bottom: 2rem;
    }
    .prediction-card {
        border-radius: 16px; padding: 1.5rem;
        color: white; text-align: center; margin: 1rem 0;
    }
    .prediction-label {
        font-size: 2rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
    }
    .confidence-text { font-size: 1.1rem; opacity: 0.9; margin-top: 0.4rem; }
    .metric-box {
        background: #f8f9fa; border-left: 4px solid #667eea;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
    }
    .metric-label { font-size: 0.8rem; color: #888; font-weight: 500; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    .warning-box {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #856404;
    }
    .footer {
        text-align: center; font-size: 0.75rem; color: #aaa;
        margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS  —  update these after training
# ─────────────────────────────────────────────
IMAGE_SIZE = (224, 224)

# Your Google Drive file ID (from the share link you provided)
MODEL_DRIVE_ID = "1YLUblgBMrSgZZKSPGEbF0FpzKQpCW-Mt"
MODEL_PATH     = "best_model.keras"

# Built-in class labels (matches your training CLASS_NAMES order)
DEFAULT_CLASS_LABELS = {
    "0": "glioma",
    "1": "meningioma",
    "2": "notumor",
    "3": "other",
    "4": "pituitary"
}

# Update these numbers after your training finishes
DEFAULT_MODEL_INFO = {
    "overall_accuracy": 0.93,
    "macro_auc":        0.98,
    "kappa":            0.91,
    "per_class_accuracy": {
        "glioma":     0.95,
        "meningioma": 0.90,
        "notumor":    0.98,
        "other":      0.88,
        "pituitary":  0.96
    }
}

CLASS_COLORS = {
    'glioma':     '#E74C3C',
    'meningioma': '#3498DB',
    'notumor':    '#2ECC71',
    'other':      '#F39C12',
    'pituitary':  '#9B59B6'
}

CLASS_INFO = {
    'glioma': {
        'icon': '🔴',
        'desc': 'A tumor arising from glial cells in the brain or spine. '
                'Ranges from slow-growing (grade I) to aggressive (grade IV / Glioblastoma).',
        'urgency': 'high'
    },
    'meningioma': {
        'icon': '🔵',
        'desc': 'A tumor forming on the meninges surrounding the brain and spinal cord. '
                'Usually benign and slow-growing.',
        'urgency': 'medium'
    },
    'notumor': {
        'icon': '🟢',
        'desc': 'No tumor detected. The MRI scan appears normal.',
        'urgency': 'low'
    },
    'other': {
        'icon': '🟡',
        'desc': "Abnormality detected that doesn't fit primary categories. "
                "May include rare tumor types. Specialist evaluation recommended.",
        'urgency': 'medium'
    },
    'pituitary': {
        'icon': '🟣',
        'desc': 'A tumor in the pituitary gland at the base of the brain. '
                'Most are benign adenomas that respond well to treatment.',
        'urgency': 'medium'
    }
}

URGENCY_STYLE = {
    'high':   ('⚠️ Requires urgent medical attention', '#ffebee', '#c62828'),
    'medium': ('ℹ️ Consult a specialist soon',          '#e8f4f8', '#1565c0'),
    'low':    ('✅ No immediate concern detected',       '#e8f5e9', '#2e7d32')
}

# ─────────────────────────────────────────────
# MODEL LOADING WITH DRIVE DOWNLOAD
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_everything():
    """Download model from Drive on first run, then load it."""

    # ── Download model if not present ─────────
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Google Drive (~500 MB) — first run only..."):
            try:
                url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None, None, None

        if not os.path.exists(MODEL_PATH):
            st.error("Download failed — file not found after download attempt.")
            return None, None, None

    # ── Load model ────────────────────────────
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None

    # ── Class labels ──────────────────────────
    if os.path.exists("class_labels.json"):
        with open("class_labels.json") as f:
            index_to_class = json.load(f)
    else:
        index_to_class = DEFAULT_CLASS_LABELS

    # ── Model info ────────────────────────────
    if os.path.exists("model_info.json"):
        with open("model_info.json") as f:
            model_info = json.load(f)
    else:
        model_info = DEFAULT_MODEL_INFO

    return model, index_to_class, model_info

# ─────────────────────────────────────────────
# PREPROCESSING  (must match training exactly)
# ─────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1]        # RGB → BGR
    arr[..., 0] -= 103.939
    arr[..., 1] -= 116.779
    arr[..., 2] -= 123.68
    return np.expand_dims(arr, 0)

# ─────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────
def make_gradcam(model, img_array, last_conv='block5_conv3'):
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            pred_idx        = tf.argmax(preds[0])
            class_channel   = preds[:, pred_idx]
        grads        = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except:
        return None

def overlay_heatmap(original_img: Image.Image, heatmap: np.ndarray, alpha=0.4):
    hm_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize(IMAGE_SIZE))
    colored    = np.uint8(mpl_cm.get_cmap('jet')(hm_resized)[:, :, :3] * 255)
    orig       = np.array(original_img.convert('RGB').resize(IMAGE_SIZE), dtype=np.float32)
    overlay    = np.uint8(orig * (1 - alpha) + colored * alpha)
    return Image.fromarray(overlay), Image.fromarray(colored)

# ─────────────────────────────────────────────
# CONFIDENCE BAR CHART
# ─────────────────────────────────────────────
def plot_confidence_bars(probs, index_to_class):
    labels = [index_to_class[str(i)] for i in range(len(probs))]
    colors = [CLASS_COLORS.get(l, '#888') for l in labels]
    pairs  = sorted(zip(probs, labels, colors), reverse=True)
    probs_s, labels_s, colors_s = zip(*pairs)

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(labels_s, [p * 100 for p in probs_s],
                   color=colors_s, edgecolor='white', linewidth=0.8)
    ax.set_xlabel('Confidence (%)', fontsize=9)
    ax.set_xlim([0, 115])
    ax.tick_params(axis='y', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, prob in zip(bars, probs_s):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob*100:.1f}%', va='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.30, 0.99, 0.50, 0.01,
        help="Predictions below this show a low-confidence warning."
    )
    show_gradcam   = st.checkbox("Show GradCAM Heatmap", value=True)
    show_all_probs = st.checkbox("Show All Class Probabilities", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    mi = DEFAULT_MODEL_INFO
    if os.path.exists("model_info.json"):
        with open("model_info.json") as f:
            mi = json.load(f)

    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-label'>Overall Accuracy</div>
        <div class='metric-value'>{mi.get('overall_accuracy', 0)*100:.1f}%</div>
    </div>
    <div class='metric-box'>
        <div class='metric-label'>Macro AUC-ROC</div>
        <div class='metric-value'>{mi.get('macro_auc', 0):.4f}</div>
    </div>
    <div class='metric-box'>
        <div class='metric-label'>Cohen's Kappa</div>
        <div class='metric-value'>{mi.get('kappa', 0):.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Per-Class Accuracy**")
    for cls, acc in mi.get('per_class_accuracy', {}).items():
        color = CLASS_COLORS.get(cls, '#888')
        icon  = CLASS_INFO.get(cls, {}).get('icon', '')
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:3px 0;font-size:0.85rem;'>"
            f"<span style='color:{color};font-weight:600;'>{icon} {cls}</span>"
            f"<span style='font-weight:700;'>{acc*100:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    for cls, cinfo in CLASS_INFO.items():
        st.markdown(f"{cinfo['icon']} **{cls.capitalize()}**")

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown(
    "<div class='main-header'>🧠 Brain Tumor MRI Classifier</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='sub-header'>Powered by VGG16 + Squeeze-and-Excitation Attention · Two-Phase Fine-Tuning</div>",
    unsafe_allow_html=True
)

# Load model (downloads from Drive on first run)
model, index_to_class, model_info = load_everything()

if model is None:
    st.error("""
    ❌ **Model could not be loaded.**

    **Fix checklist:**
    1. Open your Drive share link and confirm it is set to **"Anyone with the link"**
    2. Confirm the File ID in `app.py` matches:
       `1YLUblgBMrSgZZKSPGEbF0FpzKQpCW-Mt`
    3. Check `requirements.txt` includes `gdown>=4.7.1`
    4. On Streamlit Cloud, go to **Settings → Secrets** — no secrets needed for public Drive files
    """)
    st.stop()

st.success("✅ Model loaded and ready!")
st.markdown("---")

# Upload
uploaded = st.file_uploader(
    "📤 Upload a Brain MRI Scan (JPG / PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a T1-weighted or T1-contrast enhanced brain MRI."
)

if uploaded is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📤 Step 1**\nUpload a brain MRI image (JPG / PNG)")
    with col2:
        st.info("**⚡ Step 2**\nModel classifies tumor type in seconds")
    with col3:
        st.info("**📋 Step 3**\nReview prediction, confidence & GradCAM")

    st.markdown("""
    <div class='warning-box'>
    ⚕️ <strong>Medical Disclaimer:</strong> This tool is for research and educational
    purposes only. It is <strong>not</strong> a substitute for professional medical diagnosis.
    Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)

else:
    original_img = Image.open(uploaded)

    with st.spinner("🔍 Analysing MRI scan..."):
        processed  = preprocess_image(original_img)
        probs      = model.predict(processed, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = index_to_class[str(pred_idx)]
        confidence = float(probs[pred_idx])

    # ── PREDICTION CARD ───────────────────────
    cinfo   = CLASS_INFO[pred_class]
    urgency = URGENCY_STYLE[cinfo['urgency']]
    color   = CLASS_COLORS[pred_class]

    st.markdown(f"""
    <div class='prediction-card'
         style='background:linear-gradient(135deg,{color}dd,{color}66);'>
        <div style='font-size:2.5rem;'>{cinfo['icon']}</div>
        <div class='prediction-label'>{pred_class}</div>
        <div class='confidence-text'>Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='background:{urgency[1]};color:{urgency[2]};"
        f"padding:0.6rem 1rem;border-radius:8px;"
        f"font-size:0.9rem;font-weight:500;margin-bottom:1rem;'>"
        f"{urgency[0]}</div>",
        unsafe_allow_html=True
    )

    if confidence < confidence_threshold:
        st.warning(
            f"⚠️ **Low confidence ({confidence*100:.1f}%)** — below your threshold "
            f"of {confidence_threshold*100:.0f}%. "
            "Consider uploading a clearer scan or consulting a specialist."
        )

    with st.expander("📖 About this prediction", expanded=True):
        st.markdown(f"**{cinfo['icon']} {pred_class.capitalize()}:** {cinfo['desc']}")

    st.markdown("---")

    # ── GRADCAM / IMAGES ──────────────────────
    if show_gradcam:
        with st.spinner("🎨 Generating GradCAM..."):
            heatmap = make_gradcam(model, processed)

        if heatmap is not None:
            overlay_img, heat_img = overlay_heatmap(original_img, heatmap)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original MRI**")
                st.image(original_img.resize(IMAGE_SIZE), use_container_width=True)
            with c2:
                st.markdown("**GradCAM Heatmap**")
                st.image(heat_img, use_container_width=True)
            with c3:
                st.markdown("**Overlay**")
                st.image(overlay_img, use_container_width=True)
            st.caption("🔴 Red/yellow regions = where the model focused to make this prediction.")
        else:
            st.image(original_img.resize(IMAGE_SIZE), caption="Original MRI", width=300)
    else:
        st.image(original_img.resize(IMAGE_SIZE), caption="Uploaded MRI", width=300)

    # ── PROBABILITIES ─────────────────────────
    if show_all_probs:
        st.markdown("---")
        st.markdown("**📊 All Class Probabilities**")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = plot_confidence_bars(probs, index_to_class)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            sorted_preds = sorted(
                [(index_to_class[str(i)], float(probs[i])) for i in range(len(probs))],
                key=lambda x: x[1], reverse=True
            )
            for cls_name, prob in sorted_preds:
                is_top = cls_name == pred_class
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;padding:5px 0;'>"
                    f"<span>{CLASS_INFO[cls_name]['icon']}</span>"
                    f"<span style='flex:1;font-weight:{'700' if is_top else '400'};"
                    f"color:{'#1a1a2e' if is_top else '#555'};'>{cls_name}</span>"
                    f"<span style='font-weight:700;color:{CLASS_COLORS[cls_name]};'>"
                    f"{prob*100:.2f}%</span></div>",
                    unsafe_allow_html=True
                )

    # ── DOWNLOAD REPORT ───────────────────────
    st.markdown("---")
    mi_data = model_info or DEFAULT_MODEL_INFO
    report  = f"""BRAIN TUMOR MRI CLASSIFICATION REPORT
======================================
Predicted Class  : {pred_class.upper()}
Confidence       : {confidence*100:.2f}%
Urgency Level    : {cinfo['urgency'].upper()}

All Class Probabilities:
{chr(10).join(f"  {index_to_class[str(i)]:<14}: {probs[i]*100:.2f}%" for i in range(len(probs)))}

Model Performance Metrics:
  Overall Accuracy : {mi_data.get('overall_accuracy', 0)*100:.2f}%
  Macro AUC-ROC    : {mi_data.get('macro_auc', 0):.4f}
  Cohen's Kappa    : {mi_data.get('kappa', 0):.4f}

Per-Class Accuracy:
{chr(10).join(f"  {cls:<14}: {acc*100:.2f}%" for cls, acc in mi_data.get('per_class_accuracy', {}).items())}

DISCLAIMER: For research and educational use only.
Not a substitute for professional medical diagnosis.
"""
    st.download_button(
        "📥 Download Report (.txt)",
        data=report,
        file_name=f"brain_tumor_{pred_class}_report.txt",
        mime="text/plain"
    )

st.markdown(
    "<div class='footer'>"
    "Built with TensorFlow · VGG16 + SE Attention · Two-Phase Fine-Tuning<br>"
    "⚕️ For research and educational purposes only — not for clinical diagnosis."
    "</div>",
    unsafe_allow_html=True
)

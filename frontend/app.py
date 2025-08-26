import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

epsilon = st.slider("Epsilon (FGSM/PGD)", 0.0, 0.3, 0.03)

# Session state for reset functionality
if "reset" not in st.session_state:
    st.session_state.reset = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "ResNet18"
if "attack_type" not in st.session_state:
    st.session_state.attack_type = "FGSM"
if "epsilon" not in st.session_state:
    st.session_state.epsilon = 0.03
if "steps" not in st.session_state:
    st.session_state.steps = 10
if "kernel_size" not in st.session_state:
    st.session_state.kernel_size = 3
if "noise_level" not in st.session_state:
    st.session_state.noise_level = 0.05

st.title("Adversarial Attacks on Pretrained CV Models")

if st.button("Reset"):
    st.session_state.uploaded_file = None
    st.session_state.model_name = "ResNet18"
    st.session_state.attack_type = "FGSM"
    st.session_state.epsilon = 0.03
    st.session_state.steps = 10
    st.session_state.kernel_size = 3
    st.session_state.noise_level = 0.05
    st.experimental_rerun()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
model_name = st.selectbox("Select Model", ["ResNet18", "EfficientNet_B0", "MobileNetV2"], index=["ResNet18", "EfficientNet_B0", "MobileNetV2"].index(st.session_state.model_name))
attack_type = st.selectbox("Select Attack", ["FGSM", "PGD", "GaussianBlur", "SaltPepper", "Patch"], index=["FGSM", "PGD", "GaussianBlur", "SaltPepper", "Patch"].index(st.session_state.attack_type))
epsilon = st.slider("Epsilon (FGSM/PGD)", 0.0, 0.3, st.session_state.epsilon)
steps = st.slider("Steps (PGD)", 1, 50, st.session_state.steps)
kernel_size = st.slider("Kernel Size (Blur)", 1, 15, st.session_state.kernel_size)
noise_level = st.slider("Noise Level (Salt & Pepper)", 0.0, 0.2, st.session_state.noise_level)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": buf}
    predict_data = {"model_name": model_name}
    pred_resp = requests.post(f"{API_URL}/predict/", files=files, data=predict_data)
    orig_preds = pred_resp.json()
    st.subheader("Original Predictions")
    st.write(orig_preds)
    attack_data = {
        "model_name": model_name,
        "attack_type": attack_type,
        "epsilon": epsilon,
        "steps": steps,
        "kernel_size": kernel_size,
        "noise_level": noise_level
    }
    buf.seek(0)
    files = {"file": buf}
    if st.button("Run Attack"):
        attack_resp = requests.post(f"{API_URL}/attack/", files=files, data=attack_data)
        result = attack_resp.json()
        adv_image_b64 = result["adv_image"]
        adv_image = Image.open(io.BytesIO(base64.b64decode(adv_image_b64)))
        st.image(adv_image, caption="Adversarial Image", use_column_width=True)
        st.subheader("Adversarial Predictions")
        st.write(result["adversarial"])
        # Bar chart comparison
        orig_probs = [p["probability"] for p in orig_preds]
        adv_probs = [p["probability"] for p in result["adversarial"]]
        classes = [p["class"] for p in orig_preds]
        fig, ax = plt.subplots()
        ax.bar(classes, orig_probs, alpha=0.5, label="Original")
        ax.bar(classes, adv_probs, alpha=0.5, label="Adversarial")
        ax.set_ylabel("Probability")
        ax.legend()
        st.pyplot(fig)

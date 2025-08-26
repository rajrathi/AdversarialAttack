import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

# Configure page layout
st.set_page_config(
    page_title="Adversarial Attacks Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF5252;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.3);
    }
    .attack-button > button {
        background-color: #4ECDC4;
        color: white;
    }
    .attack-button > button:hover {
        background-color: #26C6DA;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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


st.title("🛡️ Adversarial Attacks on Pretrained CV Models")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("🎯 Attack Configuration")
    
    # Model selection
    st.subheader("🤖 Model Selection")
    model_name = st.selectbox(
        "Choose Model", 
        ["ResNet18", "EfficientNet_B0", "MobileNetV2"], 
        index=["ResNet18", "EfficientNet_B0", "MobileNetV2"].index(st.session_state.model_name),
        help="Select the pretrained model to attack"
    )
    
    # Attack type selection
    st.subheader("⚔️ Attack Type")
    attack_type = st.selectbox(
        "Choose Attack", 
        ["FGSM", "PGD", "GaussianBlur", "SaltPepper", "Patch"], 
        index=["FGSM", "PGD", "GaussianBlur", "SaltPepper", "Patch"].index(st.session_state.attack_type),
        help="Select the type of adversarial attack"
    )
    
    # Attack parameters
    st.subheader("🔧 Parameters")
    
    # Show relevant parameters based on attack type
    if attack_type in ["FGSM", "PGD"]:
        epsilon = st.slider(
            "Epsilon", 
            0.0, 0.3, st.session_state.epsilon, 
            key="epsilon_slider",
            help="Perturbation strength"
        )
    else:
        epsilon = st.session_state.epsilon
        
    if attack_type == "PGD":
        steps = st.slider(
            "Steps", 
            1, 50, st.session_state.steps, 
            key="steps_slider",
            help="Number of PGD iterations"
        )
    else:
        steps = st.session_state.steps
        
    if attack_type == "GaussianBlur":
        kernel_size = st.slider(
            "Kernel Size", 
            1, 15, st.session_state.kernel_size, 
            key="kernel_slider",
            help="Blur kernel size"
        )
    else:
        kernel_size = st.session_state.kernel_size
        
    if attack_type == "SaltPepper":
        noise_level = st.slider(
            "Noise Level", 
            0.0, 0.2, st.session_state.noise_level, 
            key="noise_slider",
            help="Amount of salt and pepper noise"
        )
    else:
        noise_level = st.session_state.noise_level
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset All", help="Reset to default values"):
        st.session_state.uploaded_file = None
        st.session_state.model_name = "ResNet18"
        st.session_state.attack_type = "FGSM"
        st.session_state.epsilon = 0.03
        st.session_state.steps = 10
        st.session_state.kernel_size = 3
        st.session_state.noise_level = 0.05
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to test adversarial attacks"
    )
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

if uploaded_file:
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Original Image", use_column_width=True)
        
        # Show original predictions
        with st.spinner("🔍 Getting predictions..."):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            files = {"file": buf}
            predict_data = {"model_name": model_name}
            try:
                pred_resp = requests.post(f"{API_URL}/predict/", files=files, data=predict_data)
                orig_preds = pred_resp.json()
                
                st.subheader("🎯 Original Predictions")
                for i, pred in enumerate(orig_preds[:3]):  # Show top 3
                    st.markdown(f"**{i+1}.** {pred['class']}")
                    st.progress(pred['probability'])
                    st.caption(f"Confidence: {pred['probability']:.3f}")
                    
            except requests.exceptions.RequestException:
                st.error("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
                st.stop()
    
    with col2:
        st.subheader("⚔️ Run Adversarial Attack")
        
        # Attack configuration display
        st.info(f"""
        **Current Configuration:**
        - Model: {model_name}
        - Attack: {attack_type}
        - Parameters: {
            f"ε={epsilon}" if attack_type in ["FGSM", "PGD"] else
            f"Steps={steps}" if attack_type == "PGD" else
            f"Kernel={kernel_size}" if attack_type == "GaussianBlur" else
            f"Noise={noise_level}" if attack_type == "SaltPepper" else
            "Default patch"
        }
        """)
        
        # Attack button with custom styling
        attack_clicked = st.button("🚀 Launch Attack", key="attack_btn", help="Generate adversarial example")
        
        if attack_clicked:
            with st.spinner("🛡️ Generating adversarial example..."):
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
                
                try:
                    attack_resp = requests.post(f"{API_URL}/attack/", files=files, data=attack_data)
                    result = attack_resp.json()
                    
                    if "error" in result:
                        st.error(f"❌ Attack failed: {result['error']}")
                    else:
                        # Display adversarial image
                        adv_image_b64 = result["adv_image"]
                        adv_image = Image.open(io.BytesIO(base64.b64decode(adv_image_b64)))
                        st.image(adv_image, caption="⚔️ Adversarial Image", use_column_width=True)
                        
                        # Show adversarial predictions
                        st.subheader("🎯 Adversarial Predictions")
                        adv_preds = result["adversarial"]
                        for i, pred in enumerate(adv_preds[:3]):  # Show top 3
                            st.markdown(f"**{i+1}.** {pred['class']}")
                            st.progress(pred['probability'])
                            st.caption(f"Confidence: {pred['probability']:.3f}")
                            
                except requests.exceptions.RequestException:
                    st.error("❌ Attack request failed. Check backend server.")

    # Comparison section
    if uploaded_file and 'orig_preds' in locals() and attack_clicked and 'result' in locals() and "error" not in result:
        st.markdown("---")
        st.subheader("📊 Prediction Comparison")
        
        # Create comparison chart
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Original Predictions**")
            orig_classes = [p["class"][:15] + "..." if len(p["class"]) > 15 else p["class"] for p in orig_preds[:5]]
            orig_probs = [p["probability"] for p in orig_preds[:5]]
            
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            bars1 = ax1.barh(orig_classes, orig_probs, color='#4ECDC4', alpha=0.8)
            ax1.set_xlabel("Probability")
            ax1.set_title("Original")
            ax1.set_xlim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars1, orig_probs):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col_chart2:
            st.markdown("**Adversarial Predictions**")
            adv_classes = [p["class"][:15] + "..." if len(p["class"]) > 15 else p["class"] for p in adv_preds[:5]]
            adv_probs = [p["probability"] for p in adv_preds[:5]]
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            bars2 = ax2.barh(adv_classes, adv_probs, color='#FF6B6B', alpha=0.8)
            ax2.set_xlabel("Probability")
            ax2.set_title("Adversarial")
            ax2.set_xlim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars2, adv_probs):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Attack success metrics
        st.markdown("---")
        st.subheader("📈 Attack Analysis")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            orig_top1 = orig_preds[0]["class"]
            adv_top1 = adv_preds[0]["class"]
            success = "✅ Success" if orig_top1 != adv_top1 else "❌ Failed"
            st.metric("Attack Status", success)
        
        with col_metric2:
            conf_drop = orig_preds[0]["probability"] - adv_preds[0]["probability"]
            st.metric("Confidence Drop", f"{conf_drop:.3f}")
        
        with col_metric3:
            ranking_change = "Changed" if orig_top1 != adv_top1 else "Unchanged"
            st.metric("Top Prediction", ranking_change)

else:
    st.info("👆 Please upload an image to start the adversarial attack demo!")
    
    # Show example images or instructions
    st.markdown("""
    ### 🚀 How to use this demo:
    1. **Upload an image** using the file uploader
    2. **Choose a model** from the sidebar (ResNet18, EfficientNet, MobileNet)
    3. **Select an attack type** and adjust parameters
    4. **Click "Launch Attack"** to generate adversarial examples
    5. **Compare results** to see how the attack affects predictions
    
    ### 🛡️ Available Attacks:
    - **FGSM**: Fast Gradient Sign Method - single step attack
    - **PGD**: Projected Gradient Descent - iterative attack  
    - **Gaussian Blur**: Simple image blurring
    - **Salt & Pepper**: Random noise injection
    - **Adversarial Patch**: Overlay attack patch
    """)

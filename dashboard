import streamlit as st
import requests
from PIL import Image
import io

# --- KONFIGURATION ---
API_URL = "https://74.220.51.0/24/predict"  # Ersätt med din faktiska Render-URL!

CLASSES = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street', 'Desert']
CLASS_EMOJIS = {
    'Buildings': '🏢', 'Forest': '🌲', 'Glacier': '🏔️',
    'Mountain': '⛰️', 'Sea': '🌊', 'Street': '🛣️', 'Desert': '🏜️'
}

# --- SIDKONFIGURATION ---
st.set_page_config(page_title="CNN Image Classifier", layout="wide")

st.title("🖼️ CNN Image Classification Dashboard")
st.markdown("**Model:** Custom CNN | **Classes:** 7 natural scene categories | **Backend:** Self-hosted API")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Model Overview")
    st.markdown("""
    **Framework:** PyTorch  
    **Architecture:** Custom CNN (2 conv layers)  
    **Input Size:** 128×128 pixels  
    **Classes:** 7 categories  
    **Test Accuracy:** 93.3%
    """)
    
    st.header("🏗️ Architecture")
    st.code("""
Conv2d(3→16, 3×3) + ReLU
MaxPool2d(2×2)
Conv2d(16→32, 3×3) + ReLU
MaxPool2d(2×2)
Flatten
Linear(32×32×32→128) + ReLU
Dropout(0.3)
Linear(128→7)
    """)
    
    st.header("📈 Training Metrics")
    st.metric("Training Accuracy", "93.4%")
    st.metric("Test Accuracy", "93.3%")
    st.metric("Loss Function", "CrossEntropyLoss")
    st.metric("Optimizer", "Adam")
    st.metric("Batch Size", "32")

# --- HUVUDINNEHÅLL ---
st.subheader("📤 Upload Image for Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("🔮 Analyzing image..."):
            # Skicka till API
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(API_URL, files=files, timeout=30)
                result = response.json()
                
                predicted = result["predicted_class"]
                confidence = result["confidence"]
                all_probs = result["all_probabilities"]
                
                # Huvudresultat
                emoji = CLASS_EMOJIS.get(predicted, '❓')
                st.success(f"**Prediction:** {emoji} {predicted}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Alla sannolikheter
                st.subheader("📊 Class Probabilities")
                for cls in CLASSES:
                    prob = all_probs.get(cls, 0)
                    bar_color = "🟢" if cls == predicted else "⚪"
                    st.progress(float(prob) / 100, text=f"{bar_color} {CLASS_EMOJIS[cls]} {cls}: {prob:.2f}%")
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
                st.info("Make sure the API server is running!")

# --- DATASET INFO ---
st.divider()
st.subheader("📚 Dataset Information")
st.markdown("""
This model was trained on the **Intel Image Classification dataset** containing ~14,000 natural scene images 
evenly distributed across 7 categories. Images were split into training and test sets for fair evaluation.
""")

cols = st.columns(7)
for i, cls in enumerate(CLASSES):
    with cols[i]:
        st.metric(f"{CLASS_EMOJIS[cls]} {cls}", "2,000+ images")

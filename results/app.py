import sys
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd

# Set up paths so we can import from src and backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models import get_model
from backend.app.utils import CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = PROJECT_ROOT / "results" / "saved_models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def extract_model_name(filename):
    # e.g., best_resnet50.pth -> resnet50
    name = filename.replace("best_", "").replace(".pth", "")
    return name

def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint

@st.cache_resource
def load_selected_model(model_filename):
    model_name = extract_model_name(model_filename)
    model = get_model(model_name, freeze_backbone=False, use_pretrained_weights=False)
    
    checkpoint_path = MODELS_DIR / model_filename
    state_dict = _load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict(image, model):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    
    predicted_class = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_class].item())
    
    probabilities = {
        class_name: float(probs[i].item())
        for i, class_name in enumerate(CLASS_NAMES)
    }
    
    return predicted_class, confidence, probabilities

st.set_page_config(page_title="APTOS Models Testing", layout="wide")
st.title("APTOS 2019 Blindness Detection - Model Testing")

# Find available models
if MODELS_DIR.exists():
    available_models = [f.name for f in MODELS_DIR.iterdir() if f.suffix == ".pth"]
else:
    available_models = []

if not available_models:
    st.error(f"No model checkpoints found in {MODELS_DIR}. Please train a model first.")
    st.stop()

st.sidebar.header("Settings")
selected_model_file = st.sidebar.selectbox("Select Model", available_models)

if selected_model_file:
    try:
        with st.spinner("Loading model..."):
            model = load_selected_model(selected_model_file)
        st.sidebar.success(f"Loaded {selected_model_file} successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()

st.header("Upload Image")
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            predict_button = st.button("Predict")
            
        with col2:
            if predict_button:
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, probabilities = predict(image, model)
                    
                st.success("Prediction complete!")
                
                st.subheader("Results")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Predicted Class", f"{predicted_class} - {CLASS_NAMES[predicted_class]}")
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                with res_col2:
                    # Plot probabilities
                    df = pd.DataFrame({
                        "Class": CLASS_NAMES,
                        "Probability": [probabilities[c] for c in CLASS_NAMES]
                    })
                    st.bar_chart(df.set_index("Class"))
                    
    except Exception as e:
        st.error(f"Error processing image: {e}")

import streamlit as st
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from gtts import gTTS
import tempfile

st.set_page_config(page_title="Hot Dogs!", page_icon="🌭")
st.header("Hot Dogs!")

# -------------------------------
# 1. Load pretrained ResNet50
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 2. File uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Prepare image for model
    input_tensor = preprocess(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    # ImageNet index 934 = "hotdog"
    hotdog_prob = probs[0, 934].item()
    is_hotdog = hotdog_prob > 0.5

    label = "HOTDOG 🌭" if is_hotdog else "NOT HOTDOG 🙅"
    st.subheader(label)
    st.write(f"Hotdog probability: {hotdog_prob:.2%}")

    # Audio output
    text = f"This is a {label.lower()}"
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
else:
    st.info("Upload an image to classify.")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("model.h5")

st.title("🧠 Brain Tumor Detection")
st.markdown("Upload a brain MRI image to detect **Tumor** or **No Tumor**.")

uploaded_file = st.file_uploader("📤 Upload MRI image", type=["jpg", "png", "jpeg"])

def preprocess_image(image, target_size=64):
    image = image.resize((target_size, target_size))
    img_array = np.array(image)
    img_array = img_array / 255.0
    return img_array.reshape(1, target_size, target_size, 3)

def predict(image):
    pred = model.predict(image)
    confidence = np.max(pred)
    result = "Tumor" if np.argmax(pred) == 1 else "No Tumor"
    return result, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        processed = preprocess_image(image)
        prediction, confidence = predict(processed)
        st.success(f"🧪 Prediction: **{prediction}**")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

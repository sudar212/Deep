import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os
from tensorflow.keras.models import load_model

# Download model from Google Drive if not exists
model_path = "brain_tumor_classifier.h5"
if not os.path.exists(model_path):
    file_id = "1wldsM6aVGXagXNmpbKA6liT0HXH6wQFB"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load model
model = load_model(model_path)

# Streamlit UI
st.title("🧠 Brain Tumor Detection App")
st.markdown("Upload an MRI scan image to check for brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Output
    if prediction[0][0] > 0.5:
        st.error("🧠 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

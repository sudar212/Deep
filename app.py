import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title
st.title("🧠 Brain Tumor Detection App")
st.markdown("Upload an MRI scan image to check for brain tumor.")

# Load model
model = load_model("brain_tumor_classifier.h5")

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

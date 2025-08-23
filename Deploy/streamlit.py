import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


st.set_page_config(page_title="Crack Detection System", layout="wide")

MODEL_PATH = r"D:\SUDIP\mlcrackdetection\MLTF\Deploy\mlcrackdetection.keras"
TARGET_SIZE = (200, 200) 
CHANNELS = 1 


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        # Verify input shape
        if model.input_shape[1:3] != TARGET_SIZE:
            st.error(
                f"Model expects {model.input_shape[1:3]} input, but code is set to {TARGET_SIZE}")
            return None

        st.success(
            f"✅ Model loaded successfully! Input shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None


model = load_model()

# --- Preprocess ---


def preprocess_image(image):
    try:
        img = np.array(image)

        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, TARGET_SIZE)

        # Normalize to [-1,1]
        img = img / 255.0
        img = img * 2.0 - 1.0

        # Expand dimensions: (200,200,1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)  # batch

        return img
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None

# --- Prediction ---


def predict_image(model, image):
    try:
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None, None, None

        prediction = model.predict(processed_img, verbose=0)

        confidence = float(prediction[0][0])
        label = "Cracked" if confidence > 0.5 else "Uncracked"
        confidence = max(confidence, 1 - confidence)

        return label, confidence, processed_img[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None


# --- UI ---
st.title("🧱 Automated Crack Detection System")
st.write("Upload an image for instant crack detection (200×200 grayscale).")

with st.sidebar:
    st.header("Settings")
    image_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"])
    if model:
        st.write(f"Model input size: {TARGET_SIZE}")
        st.write(f"Channels: {CHANNELS}")
        st.write(f"Output: Sigmoid (binary classification)")

# --- Main Processing ---
if image_file and model:
    try:
        image = Image.open(image_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        label, confidence, processed_img = predict_image(model, image)

        if label:
            with col2:
                st.subheader("Analysis Results")
                status_color = "red" if label == "Cracked" else "green"
                st.markdown(
                    f"<h3 style='color:{status_color}'>Status: {label}</h3>", unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.2%}")

                # Processed image for visualization
                display_img = ((processed_img.squeeze() + 1.0) /
                               2.0 * 255).astype(np.uint8)
                st.image(
                    display_img, caption=f"Processed Image ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})", use_container_width=True)

    except Exception as e:
        st.error(f"Processing error: {str(e)}")

elif not model:
    st.warning("⚠ Model not loaded - check path")
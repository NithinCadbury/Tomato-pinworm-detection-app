import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tempfile

# Set up page title
st.set_page_config(page_title="Tuta Absoluta Detector", layout="centered")
st.title("🪰 Tuta Absoluta Detection App")

# Load model
@st.cache_resource
def load_resnet_model():
    model = load_model("ResNet50.h5")
    return model

model = load_resnet_model()

# Class labels
class_names = ['NonTarget', 'Tuta_Adult', 'Tuta_Leaf']

# Preprocess image for ResNet50
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Remove alpha if present
    img_array = preprocess_input(img_array.astype(np.float32))
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict(image: Image.Image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[class_index], confidence

# Capture from webcam
def capture_from_camera():
    cap = cv2.VideoCapture(0)
    st.info("Press 'Space' to capture image. Press 'q' to quit.")
    img_captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        cv2.imshow("Camera - Press Space to Capture", frame)
        key = cv2.waitKey(1)

        if key == ord(' '):
            img_captured = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if img_captured is not None:
        img_rgb = cv2.cvtColor(img_captured, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    return None

# Choose input method
option = st.radio("Choose image source:", ('Upload an Image', 'Use Camera'))

image = None

if option == 'Upload an Image':
    uploaded_file = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == 'Use Camera':
    if st.button("Open Camera"):
        image = capture_from_camera()

# Predict
if image:
    st.image(image, caption="Input Image", use_container_width=True)
    label, confidence = predict(image)
    st.success(f"🧠 Prediction: **{label}** ({confidence*100:.2f}% confidence)")

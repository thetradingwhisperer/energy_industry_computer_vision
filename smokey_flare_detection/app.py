import streamlit as st
import cv2
import numpy as np
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title('Smoke Detection Demo')
st.write("This demo shows how our smoke detection model works on the offshore platforms.")

image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if image_file is not null:
    image = load_image(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Detect Smoke'):
        # Assuming 'detect_smoke' is a function that processes the image and returns an annotated one
        processed_image = detect_smoke(image)
        st.image(processed_image, caption='Processed Image with Detected Smoke', use_column_width=True)

# Model Metrics
st.header("Model Metrics")
st.write("Here you can see the performance metrics of our smoke detection model.")
# Assuming metrics are stored in a dictionary
metrics = {'Accuracy': 0.95, 'Precision': 0.88, 'Recall': 0.90}
for metric, value in metrics.items():
    st.metric(label=metric, value=f"{value * 100}%")

# Documentation
st.header("How It Works")
st.write("""
Our smoke detection model is trained on thousands of images from oil platforms, using a deep learning framework to identify smoke patterns even in challenging conditions. By implementing this model, we aim to enhance safety measures and response times significantly.
""")

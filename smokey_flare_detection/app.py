import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import make_inference
import tempfile

def load_image(image_file):
    img = Image.open(image_file)
    img_for_model = np.array(img)
    img_for_model_bgr = cv2.cvtColor(img_for_model, cv2.COLOR_RGB2BGR)

    # Save the numpy array image to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp.name, img_for_model_bgr)

    return img, temp.name


st.title('Smokey Flare Detection Demo')
st.write("This demo shows how our smoke detection model works on the offshore flaring.")
# Documentation
st.header("How It Works")
st.write("""
Our smoke detection model is trained on thousands of images from oil platforms, 
using a deep learning framework to identify smoke patterns even in challenging conditions. 
By implementing this model, we aim to use AI to identify smokey flaring situations and consequently
trigger investigation to mitigate these events.
""")
st.divider()
st.write("To use the model, upload an image of an offshore platform and click the 'Detect' button.")

image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if image_file is not None:
    image, image_path = load_image(image_file)
    #Define 2 columns
    col1, col2 = st.columns(2)

    with col1:
        col1.header('Original Image')
        #st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(image_path, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect'):
            with col2:
                # Assuming 'detect_smoke' is a function that processes the image and returns an annotated one
                col2.header('Processed Image')
                processed_image = make_inference(image_path)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_image, 
                    caption='Processed Image with Detected Smoke', use_column_width=True)





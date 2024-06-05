import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import make_inference_normal, make_inference_ringelmaan
import tempfile

def load_image(image_file):
    img = Image.open(image_file)
    img_for_model = np.array(img)
    img_for_model_bgr = cv2.cvtColor(img_for_model, cv2.COLOR_RGB2BGR)

    # Save the numpy array image to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp.name, img_for_model_bgr)

    return img, temp.name

with st.sidebar:
    st.title('Smokey Flare Detection with Vision AI')
    st.write('1) Please upload an image of an offshore platform to detect smoke.')
    st.write('2) Choose the type of detection "normal" or "ringelmaan"')
    st.write('3) Click the "Detect" button to see the vision ai detection results.')
    st.divider()

    confidence_threshold = st.slider('Set detection confidence Threshold', 0, 100, 40)


    st.header("How It Works")
    st.markdown("""
    Our smoke detection model is trained on thousands of images from oil platforms, 
    using a deep learning framework to identify smoke patterns even in challenging conditions. 
    By implementing this model, we aim to use AI to identify smokey flaring situations and consequently
    trigger investigation to mitigate these events.
    """)
#st.subheader('Smokey Flare Detection with AI')
#st.write("This demo shows how our smoke detection model works on the offshore flaring.")
# Documentation
#st.divider()
st.logo("NOC logo.PNG")
st.image("NOC logo.PNG")
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Add two tabs for two prediction type: 1) fire and some 2) fire and different type of smoke
tab1, tab2 = st.tabs(["Normal", "Ringelmaan"])
tab1.subheader("Fire and Smoke Detection")
tab2.subheader("Fire and Ringelmaan Smoke Type Detection")

with tab1:
    st.markdown("""This tab uses a Vision AI model that is trained to detect **fire/flames and smoke** in an 
                uploaded image.""")
    
    st.markdown("""This model is trained on approximately 1000 images of offshore flare stacks with smoke and fire, and therefore
                has slightly better performance on low-resolution images.""")
    if image_file is not None:
        image, image_path = load_image(image_file)
        #Define 2 columns
        col1, col2 = st.columns(2)

        with col1:
            col1.markdown('###### Original Image')
            #st.image(image, caption='Uploaded Image', use_column_width=True)
            st.image(image_path, caption='Uploaded Image', use_column_width=True)
            #add empty space
            st.write('')
            if st.button('Detect', key='detect_normal'):
                with col2:
                    # Assuming 'detect_smoke' is a function that processes the image and returns an annotated one
                    col2.markdown('###### AI Processed Image')
                    processed_image, labels = make_inference_normal(image_path, confidence=confidence_threshold)
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_image, 
                        caption='Processed Image with Detected Smoke', use_column_width=True)
                    st.html(f'<p style="color:red;"> Results: {labels} detected in the image</p>')

with tab2:
    st.markdown("""This tab uses a Vision AI model that is trained to detect **fire/flames and 3 types of smoke: r1, r3 & r5** in an 
                uploaded image, corresponding to ringelmaan 1, 3 and 5""")

    st.markdown("""This model is trained on approximately 370 images of offshore flare stacks with smoke and fire, and therefore 
                please expect lower performance or errors on low-resolution images.""")
    if image_file is not None:
        image, image_path = load_image(image_file)
        #Define 2 columns
        col1_2, col2_2 = st.columns(2)

        with col1_2:
            col1_2.markdown('###### Original Image')
            #st.image(image, caption='Uploaded Image', use_column_width=True)
            st.image(image_path, caption='Uploaded Image', use_column_width=True)
            #add empty space
            st.write('')
            if st.button('Detect', key='detect_ringelmaan'):
                with col2_2:
                    # Assuming 'detect_smoke' is a function that processes the image and returns an annotated one
                    col2_2.markdown('###### AI Processed Image')
                    processed_image, labels = make_inference_ringelmaan(image_path, confidence=confidence_threshold)
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.image(processed_image, 
                        caption='AI Processed Image with detection', use_column_width=True)
                    st.html(f'<p style="color:red;"> Results: {labels} detected in the image</p>')



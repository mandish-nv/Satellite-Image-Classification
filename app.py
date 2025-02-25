# streamlit run app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('finalDataModel.keras')

# Define class labels
class_labels = ['agriculture', 'forest', 'urban', 'water']

# Image size (must match training input size)
img_height, img_width = 75, 75

st.title("Satellite Image Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and resize image
        img = Image.open(uploaded_file)
        img = img.resize((img_width, img_height))

        # Convert to NumPy array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Apply Image Augmentation (Same as Training)
        datagen = image.ImageDataGenerator(
            rescale=1./255
        )

        # Process image using datagen (augment the input image)
        img_preprocessed = next(datagen.flow(img_array, batch_size=1))

        # Predict
        predictions = model.predict(img_preprocessed)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_labels[predicted_class_index]
        confidence = np.max(predictions) * 100

        # Display results
        st.subheader(f"Predicted Category: **{predicted_class_name}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.image(img, caption="Original Uploaded Image", width=250)


    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

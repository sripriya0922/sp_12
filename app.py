import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
@st.cache(allow_output_mutation=True)
def load_keras_model():
    return load_model("keras_Model.h5", compile=False)
model = load_keras_model()
# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()
# Function to preprocess image
def preprocess_image(image):
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array
# Streamlit app
def main():
    st.title("Image Classifier")
    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Make prediction
        if st.button("Predict"):
            # Preprocess the image
            normalized_image_array = preprocess_image(image)
            
            # Create data array
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            # Predict
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]
            # Display prediction
            st.write(f"Prediction: {class_name}")
            st.write(f"Confidence Score: {confidence_score}")
if __name__ == "__main__":
    main()

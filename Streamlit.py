import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the trained model
model = load_model('Best_Model.keras')

# Define the labels
labels = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return labels[predicted_class], confidence

# Streamlit app
st.title('Dementia Classification')
st.write('Upload an MRI brain scan image for classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, confidence = predict(image)
    st.write(f'Prediction: {label} (Confidence: {confidence})')

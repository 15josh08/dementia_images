from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model
import io

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(io.BytesIO(file.read()))
            label, confidence = predict(image)
            return render_template('result.html', label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)

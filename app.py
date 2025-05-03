from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."
    
    file = request.files['file']
    if file.filename == '':
        return "Empty filename."

    try:
        img = Image.open(file).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        return render_template('index.html', prediction=f'Predicted Digit: {predicted_digit}')
    except Exception as e:
        return f"Error processing image: {e}"

if __name__ == '__main__':
    app.run(debug=True)

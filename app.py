from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    img = Image.open(file).convert('L')  # convert to grayscale
    img = img.resize((28, 28))  # resize to 28x28
    img_array = np.array(img) / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28)
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    return render_template('index.html', prediction=f'Predicted Digit: {digit}')

if __name__ == '__main__':
    app.run(debug=True)

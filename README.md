# Handwritten Digit Classification using ANN (MNIST Dataset)

## Project Overview

This project focuses on classifying handwritten digits from the MNIST dataset using an Artificial Neural Network (ANN). The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0 to 9). The goal of this project is to build an ANN model that can accurately predict the digit in an input image.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)
- [License](#license)

## Dataset

The dataset used in this project is the **MNIST** dataset, which consists of 28x28 pixel grayscale images of handwritten digits (0-9). Each image is labeled with the corresponding digit.

- **Training Set**: 60,000 images 
- **Test Set**: 10,000 images

## Technologies Used

- Python 3.x
- TensorFlow / Keras (for building and training the model)
- NumPy (for numerical operations)
- Matplotlib (for visualization)
- scikit-learn (for model evaluation)

## Model Architecture

The ANN used in this project consists of the following layers:

1. **Input Layer**: 784 nodes (28x28 pixels flattened into a vector)
2. **Hidden Layer 1**: 128 neurons, ReLU activation function
3. **Hidden Layer 2**: 64 neurons, ReLU activation function
4. **Output Layer**: 10 neurons (one for each digit 0-9), Softmax activation function

## Installation

To run the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/handwritten-digit-classification.git
   cd handwritten-digit-classification
   
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

## Results

After training, the model achieves an accuracy of 97.37% on the test set. The model's performance can be improved by experimenting with different architectures, optimizers, and hyperparameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






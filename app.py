import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from flask_cors import CORS # Import CORS
from skimage.transform import resize

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

MODEL_PATH = 'mnist_digit_recognizer.h5'

# --- Model Training Function ---
def train_and_save_model():
    """
    Trains a CNN model on the MNIST dataset and saves it.
    This function will only run if the model file does not exist.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_PATH}' already exists. Skipping training.")
        return

    print("Model not found. Training a new model...")

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data
    # Reshape to (28, 28, 1) for CNN input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize pixel values to 0-1
    x_train /= 255
    x_test /= 255

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    # Using a smaller number of epochs for demonstration purposes
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, validation_data=(x_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model trained and saved as '{MODEL_PATH}'")

# --- Load the trained model ---
# This will be called once when the Flask app starts
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Attempting to train a new model.")
    train_and_save_model() # Train if model not found or corrupted
    model = load_model(MODEL_PATH) # Load after training

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image, preprocesses it, and returns a digit prediction.
    Expects a JSON payload with 'image' field containing base64 encoded image data.
    """
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = request.json['image']
    # Remove the "data:image/png;base64," prefix if present
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to grayscale

        # Resize to 28x28 pixels
        # Use skimage.transform.resize for better quality resizing and anti-aliasing
        img_resized = resize(np.array(img), (28, 28), anti_aliasing=True)

        # Invert colors (MNIST digits are white on black background, uploaded might be black on white)
        # Check if the average pixel value is high (suggests white background)
        if np.mean(img_resized) > 0.5:
            img_resized = 1 - img_resized # Invert colors

        # Normalize pixel values to 0-1 (already done by resize but good to be explicit)
        # Reshape for model input: (1, 28, 28, 1)
        img_array = img_resized.reshape(1, 28, 28, 1).astype('float32')

        # Make prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        return jsonify({'prediction': int(predicted_digit)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Image processing or prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode for development

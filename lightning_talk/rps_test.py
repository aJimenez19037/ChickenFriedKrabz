import os

import tensorflow as tf
import numpy as np
import keras_cv
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def preprocess_single_image(img_path, target_size=(32,300)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(img_path):
    # Preprocess the image
    img_array = preprocess_single_image(img_path)
    
    # Get the model's prediction
    predictions = model.predict(img_array)
    
    # Decode the prediction
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map predicted class to a label (rock, paper, scissors)
    label_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    predicted_label = label_map.get(predicted_class, "Unknown")
    
    print(f"Predicted Label: {predicted_label}")
    return predicted_label


# Load the saved model using custom_objects to handle the custom layer
custom_objects = {"ImageClassifier": keras_cv.models.ImageClassifier}  # Register the custom object

# Load the saved model (use this instead of training the model every time)
model = load_model('rps_model.h5', custom_objects=custom_objects)

# Preprocess the image and predict
img_path = 'images/p1.jpg'  # Replace with the path to your image
predict_image(img_path)
img_path = 'images/p2.jpg'  # Replace with the path to your image
predict_image(img_path)
img_path = 'images/r1.jpg'  # Replace with the path to your image
predict_image(img_path)
img_path = 'images/r2.jpg'  # Replace with the path to your image
predict_image(img_path)
img_path = 'images/s1.jpg'  # Replace with the path to your image
predict_image(img_path)
img_path = 'images/s3.png'  # Replace with the path to your image
predict_image(img_path)

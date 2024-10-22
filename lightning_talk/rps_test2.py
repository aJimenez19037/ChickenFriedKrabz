import os

import tensorflow as tf
import numpy as np
import keras_cv
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt  # Importing matplotlib for displaying images
from PIL import Image, ImageOps  # Import both Image and ImageOps from PIL


def preprocess_single_image(img_path, target_size = (300,300)):  # Adjust target size if needed
    img = Image.open(img_path)  # Use PIL's Image.open to load the image
    img = img.convert('RGB')  # Ensure the image is in RGB (3 channels)
    img = ImageOps.fit(img, target_size, Image.LANCZOS)  # Resize while maintaining aspect ratio
    img = img.rotate(-90, expand=True)  # Rotate 90 degrees clockwise. This was needed for our application specifically.
    img_array = image.img_to_array(img)  # Convert to array for Keras compatibility
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
 
    return img_array

def predict_image(img_path):
    # Preprocess the image
    img_array = preprocess_single_image(img_path)
    
    print("PREDICTING MODEL ...")
    # Get the model's prediction
    predictions = model.predict(img_array)
    print("... MODEL PREDICTED")
    
    # Decode the prediction
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map predicted class to a label (rock, paper, scissors)
    label_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    predicted_label = label_map.get(predicted_class, "Unknown")
    
    print(f"Predicted Label: {predicted_label}")
    print()
    return img_array, predicted_label


# Load the saved model using custom_objects to handle the custom layer
custom_objects = {"ImageClassifier": keras_cv.models.ImageClassifier}  # Register the custom object

# Load the saved model (use this instead of training the model every time)
model = load_model('rps_model.h5', custom_objects=custom_objects)
#print(f"Model input shape: {model.input_shape}") Check what type of img the model requires. 
#Model is flexible in image size, but requires it to have 3 channels "RGB"


# Preprocess the image and predict
img_paths = [
    'images/p1.jpg',  # Replace with the path to your image
    'images/p2.jpg',
    'images/r1.jpg',
    'images/r2.jpg',
    'images/s1.jpg',
    'images/s3.png'
]

# Loop through and predict for each image
for img_path in img_paths:
    processed_img, predicted_label = predict_image(img_path)
    
    # Remove batch dimension for displaying the image
    processed_img = np.squeeze(processed_img, axis=0)

    # Ensure the image is within the correct range [0, 255] for display
    processed_img = processed_img.astype('uint8')  # Convert to uint8 if needed

    # Display the image with the predicted label
    #plt.imshow(image.load_img(img_path))
    plt.imshow(processed_img)
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')  # Hide axes for better clarity
    plt.show()


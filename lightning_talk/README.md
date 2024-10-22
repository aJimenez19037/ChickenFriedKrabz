# Keras CV Tutorial
### Members 
Antonio Jimenez, CJ Ramirez, Joshua Caratao
## Objective
Train, Save, and Test an image classifier model on rock, paper, scissors dataset
## Tutorial
### Install KerasCV and TensorFlow
    pip install --upgrade keras-cv tensorflow
### Install Keras
    pip install --upgrade keras

### Run training file and save Image Classifier Model (Optional)
After installing the necessary packages, you can run "keras_cv_demo.py." This script uses KerasCV with Keras and TensorFlow to access a Rock, Paper, Scissors dataset, preprocess and augment the images, and train a model to classify the images. A pre-trained model is already included in this repository, so running the script is optional.

If you choose to run the training file, make sure you are in the correct directory and run either command below (This may take awhile).

    python keras_cv_demo.py

OR

    python3 keras_cv_demo.py


### Run test file
After the model has been trained, run the "rps_test.py" file. This file takes in images included in our "images" directory and uses the model to classify each image as either Rock, Paper, or Scissors

    python rps_test.py

OR

    python3 rps_test.py

### Hooray, you are done!




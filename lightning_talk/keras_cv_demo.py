import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import tensorflow as tf
import keras_cv
import tensorflow_datasets as tfds
import keras

# Create a preprocessing pipeline with augmentations
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(), # randomly flip images
        keras_cv.layers.RandAugment(value_range=(0, 255)),  # applies random augments
        keras_cv.layers.CutMix(), # stiches parts of different images onto an image
    ],
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES) # one hot encode
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    if augment:
        outputs = augmenter(outputs) # augment outputs 
    return outputs['images'], outputs['labels']

train_dataset, test_dataset = tfds.load( # load in rock paper scissors dataset and split into train and test
    'rock_paper_scissors',
    as_supervised=True,
    split=['train', 'test'],
)
train_dataset = train_dataset.batch(BATCH_SIZE).map( # preprocess train data using batches of 16 images and parallelization
    lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

# Create a model using a pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(  # get keras_cv pretrained backbone - extract high quality features
    "efficientnetv2_b0_imagenet"
)
model = keras_cv.models.ImageClassifier( # cater pretained model for image classification 
    backbone=backbone,
    num_classes=NUM_CLASSES,
    activation="softmax",
)
model.compile( # select loss function, optimizer, and metric being optimized
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# Train your model
model.fit( # train dataset
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)
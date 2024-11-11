import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization


# Create a preprocessing pipeline with augmentations
BATCH_SIZE = 16
NUM_CLASSES = 1
LEARNING_RATE = 0.001
GLOBAL_CLIPNORM = 10.0
augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
    ],
)
# Set the path to your image directory
train_label_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop/Courses/Fall 2024/ME369_Python/FinalProject/AerialViewDataset/train/labels")
train_image_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop/Courses/Fall 2024/ME369_Python/FinalProject/AerialViewDataset/train/images")
test_label_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop/Courses/Fall 2024/ME369_Python/FinalProject/AerialViewDataset/test/labels")
test_image_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop/Courses/Fall 2024/ME369_Python/FinalProject/AerialViewDataset/test/images")


def load_yolo_labels(label_path, image_size=(640, 640)):
    """
    Load YOLO format labels (class_id x_center y_center width height).
    Convert them into [x_center, y_center, width, height] in normalized format.
    """
    boxes = []
    labels = []
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])  # Class ID
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert to (x_center, y_center, width, height)
            boxes.append([x_center, y_center, width, height])
            labels.append(class_id)
    
    return np.array(boxes), np.array(labels)

def load_image(image_path, target_size=(640, 640)):
    """
    Load and resize the image to the target size.
    Normalize the image to [0, 1] range.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array

    def create_dataset(image_dir, label_dir, target_size=(640, 640)):
    images = []
    bboxes = []
    class_names = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            if os.path.exists(label_path):
                # Load the image
                img = load_image(image_path, target_size)
                images.append(img)
                
                # Load the bounding boxes and labels
                boxes, labels = load_yolo_labels(label_path, target_size)
                bboxes.append(boxes)
                class_names.append(labels)
    
    images = np.array(images)
    bboxes = np.array(bboxes)
    class_names = np.array(class_names)
    
    return images, bboxes, class_names

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    if augment:
        outputs = augmenter(outputs)
    return outputs['images'], outputs['labels']

# def preprocess_data(images, labels, augment=False):
#     # Restructure labels to include 'classes' and 'boxes'
#     labels_dict = {
#         "classes": tf.one_hot(labels, NUM_CLASSES),  # Your class labels
#         "boxes": "xywh"              # Bounding box data; make sure this is defined correctly
#     }
    
#     inputs = {"images": images, "labels": labels_dict}
#     outputs = inputs
#     if augment:
#         outputs = augmenter(outputs)
#     return outputs['images'], outputs['labels']




# Set the path to your image directory
train_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop/Courses/Fall 2024/ME369_Python/FinalProject/AerialViewDataset/train")
test_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop/Courses/Fall 2024/ME369_Python/FinalProject/AerialViewDataset/test")

# Load datasets from the directory
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_path,
    batch_size=BATCH_SIZE,
    label_mode='int',  # or 'categorical' if you have multiple classes
    image_size=(640, 640)  # specify your desired image size
)

# Check if your train and test datasets have the expected label structure
for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Labels structure:", labels)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_path,
    batch_size=BATCH_SIZE,
    label_mode='int',  # or 'categorical' if you have multiple classes
    image_size=(640, 640)  # specify your desired image size
)

# Apply the preprocessing and augmentations to the datasets
# train_dataset = train_dataset.batch(BATCH_SIZE).map(
#     lambda x, y: preprocess_data(x, y, augment=True),
#         num_parallel_calls=tf.data.AUTOTUNE).prefetch(
#             tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE).map(
#     preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
#         tf.data.AUTOTUNE)

# Apply the preprocessing and augmentations to the datasets
train_dataset = train_dataset.map(
    lambda x, y: preprocess_data(x, y, augment=True),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(
    preprocess_data,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Create a model using a pretrained backbone
model = keras_cv.models.YOLOV8Detector(
    num_classes=NUM_CLASSES,
    bounding_box_format="xywh",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_m_backbone_coco"
    ),
    # fpn_depth=2
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

model.compile(
    optimizer=optimizer,
    classification_loss="binary_crossentropy", 
    box_loss="ciou"
)

# Train your model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)

model.save('AerialFPDM_YOLOmodel.h5')
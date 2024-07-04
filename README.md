# -Image-Classification-and-Object-Detection-using-Convolutional-Neural-Networks
Object Detection


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

# Assuming you have your data directories
train_data_dir = 'path_to_train_data_directory'
test_data_dir = 'path_to_test_data_directory'
object_detection_model_url = 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 10
num_classes = 10  # Number of classes for classification

# Image Classification using VGG16
def build_classification_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(base_model.input, x)
    return model

# Object Detection using TensorFlow Hub
def build_object_detection_model():
    object_detection_module = tf.keras.Sequential([
        hub.KerasLayer(object_detection_model_url)
    ])
    return object_detection_module

# Data Augmentation and Loading
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Compile and Train Classification Model
classification_model = build_classification_model()
classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classification_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator))

# Object Detection
object_detection_model = build_object_detection_model()

# Example of Object Detection on an Image
def detect_objects(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (img_height, img_width))
    input_tensor = tf.convert_to_tensor(image_resized)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = object_detection_model.predict(input_tensor)
    # Process detections and visualize results

# Example usage
image_path = 'path_to_image.jpg'
detect_objects(image_path)

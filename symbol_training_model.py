import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(folder_paths):
    images = []
    labels = []
    for class_label, folder_path in enumerate(folder_paths):
        for file_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            image = cv2.resize(image, (28, 28))
            images.append(image)
            labels.append(class_label)
    return np.array(images), np.array(labels)

# Define paths to folders containing images for each symbol
addition_folder = "../Handwritten_Digits_Recognition/symbol/add"
subtraction_folder = "../Handwritten_Digits_Recognition/symbol/sub"
multiplication_folder = "../Handwritten_Digits_Recognition/symbol/mul"
division_folder = "../Handwritten_Digits_Recognition/symbol/div"

# Create a list of all folder paths
folders = [addition_folder, subtraction_folder, multiplication_folder, division_folder]

# Load data using the folders list
train_data, train_labels = load_data(folders)

# Check class distribution
unique, counts = np.unique(train_labels, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Normalize and reshape
train_data = train_data.astype('float32') / 255.0
train_data = np.expand_dims(train_data, axis=-1)

# Split data
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

# Class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test),
    class_weight=class_weights
)

# Save
model.save('symbol_model.h5')

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
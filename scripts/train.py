# -*- coding: utf-8 -*-
"""
**1. Unzip the Dataset**

This script will extract the files into a directory named currency_data.
"""

import zipfile
import os

zip_file_path = 'Bill_dataset.zip'
extract_to_dir = 'currency_data'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_dir)
    print(f"Dataset extracted to: {extract_to_dir}")

# Verify the folders inside
print("Folders found:", os.listdir(extract_to_dir))

"""**2. Organize Test Data (Splitting)**

Now that the files are extracted, we will create the test_data folder and move a portion of images into it. This ensures that the test images are completely removed from the training set
"""

import shutil
import random

source_path = 'currency_data/Bill_dataset'
test_path = 'test_data'
denominations = ['1', '5', '10', '20'] # Based on your folder names

if not os.path.exists(test_path):
    os.makedirs(test_path)

for folder in denominations:
    # Create the same subfolder in test_data
    os.makedirs(os.path.join(test_path, folder), exist_ok=True)

    # Path to the source images
    current_folder_path = os.path.join(source_path, folder)
    images = os.listdir(current_folder_path)

    # Select 20% of images to move to the test folder
    num_to_move = int(len(images) * 0.2)
    images_to_move = random.sample(images, num_to_move)

    for img in images_to_move:
        shutil.move(os.path.join(current_folder_path, img),
                    os.path.join(test_path, folder, img))

print("Data split complete. Test folder is ready.")

"""**3. Training and Evaluation**

Finally, we use TensorFlow to train the model on the remaining images in the currency_data folder and evaluate it against the test_data folder.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data Generators (Consistent with your folder structure)
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

# Assuming the path you provided in the last run
train_data = train_gen.flow_from_directory('currency_data/Bill_dataset', target_size=(150, 150), batch_size=32)
test_data = test_gen.flow_from_directory('test_data', target_size=(150, 150), batch_size=32)

# 2. Optimized Simple CNN Model (Warning-Free Version)
model = models.Sequential([
    Input(shape=(150, 150, 3)), # Explicitly defines input to avoid layer warnings
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Training and Evaluation
model.fit(train_data, epochs=5)
test_loss, accuracy = model.evaluate(test_data)
print(f"\nFinal Accuracy on Test Set: {accuracy * 100:.2f}%")


# -*- coding: utf-8 -*-
"""
**Create a Prediction Script**

This script will take a single image path, process it to match the $150 \times 150$ input size, and print the predicted class.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_bill_value(img_path):
    # 1. Load and process the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Same normalization as training

    # 2. Load the saved model
    loaded_model = tf.keras.models.load_model('dollar_detector_model.keras')

    # 3. Predict
    predictions = loaded_model.predict(img_array)
    class_indices = {0: '1', 1: '10', 2: '20', 3: '5'} # Verify this with train_data.class_indices

    predicted_class = class_indices[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"Predicted Value: ${predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

predict_bill_value('/content/a0056_A.tif')
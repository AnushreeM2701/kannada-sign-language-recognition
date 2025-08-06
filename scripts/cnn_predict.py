#!/usr/bin/env python3
"""
CNN-based prediction functions for Kannada Sign Language Recognition
Uses the newly trained CNN models
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
IMG_SIZE = 64
NUM_CHANNELS = 3

# Load model paths
MODEL_DIR = "models"
ALPHABET_MODEL_PATH = os.path.join(MODEL_DIR, "alphabet_model.keras")
WORD_MODEL_PATH = os.path.join(MODEL_DIR, "word_model.keras")
ALPHABET_LABEL_MAP_PATH = os.path.join(MODEL_DIR, "alphabet_label_map.json")
WORD_LABEL_MAP_PATH = os.path.join(MODEL_DIR, "word_label_map.json")

# Load models and label maps
alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL_PATH)
word_model = tf.keras.models.load_model(WORD_MODEL_PATH)

with open(ALPHABET_LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    alphabet_label_map = json.load(f)
    alphabet_reverse_map = {v: k for k, v in alphabet_label_map.items()}

with open(WORD_LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    word_label_map = json.load(f)
    word_reverse_map = {v: k for k, v in word_label_map.items()}

def preprocess_image(image):
    """Preprocess image for CNN input"""
    if image is None:
        return None
    
    # Resize to CNN input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    image = image.astype("float32") / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_alphabet(frame):
    """Predict Kannada alphabet using CNN model"""
    try:
        # Preprocess the frame
        processed_image = preprocess_image(frame)
        if processed_image is None:
            return None
        
        # Make prediction
        predictions = alphabet_model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get the actual alphabet character
        predicted_alphabet = alphabet_reverse_map.get(predicted_class, "Unknown")
        
        return {
            'prediction': predicted_alphabet,
            'confidence': confidence,
            'class_id': int(predicted_class)
        }
        
    except Exception as e:
        print(f"Error in alphabet prediction: {e}")
        return None

def predict_word_sequence(frames):
    """Predict word from sequence of frames using CNN model"""
    try:
        if not frames:
            return None
        
        # Process each frame
        predictions = []
        confidences = []
        
        for frame in frames:
            processed_image = preprocess_image(frame)
            if processed_image is not None:
                pred = word_model.predict(processed_image, verbose=0)
                predicted_class = np.argmax(pred[0])
                confidence = float(np.max(pred[0]))
                
                predictions.append(predicted_class)
                confidences.append(confidence)
        
        if not predictions:
            return None
        
        # Get the most common prediction
        from collections import Counter
        most_common = Counter(predictions).most_common(1)
        if most_common:
            final_class = most_common[0][0]
            avg_confidence = np.mean([conf for pred, conf in zip(predictions, confidences) 
                                    if pred == final_class])
            
            predicted_word = word_reverse_map.get(final_class, "Unknown")
            
            return {
                'prediction': predicted_word,
                'confidence': float(avg_confidence),
                'class_id': int(final_class),
                'frame_count': len(frames)
            }
        
        return None
        
    except Exception as e:
        print(f"Error in word prediction: {e}")
        return None

def safe_predict(frame):
    """Safe prediction wrapper with error handling"""
    result = predict_alphabet(frame)
    if result is None:
        return "Error"
    return result['prediction']

# Test function
if __name__ == "__main__":
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test alphabet prediction
    alphabet_result = predict_alphabet(dummy_image)
    print(f"Alphabet test: {alphabet_result}")
    
    # Test word prediction
    frames = [dummy_image] * 5
    word_result = predict_word_sequence(frames)
    print(f"Word test: {word_result}")
    
    print("✅ CNN prediction functions loaded successfully")

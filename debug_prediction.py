#!/usr/bin/env python3
"""
Debug script to identify prediction issues in Kannada Sign Language Recognition
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.cnn_predict_fixed import predict_alphabet, predict_word_sequence
from scripts.predict_updated import preprocess_image

def debug_model_loading():
    """Debug model loading process"""
    print("=== Model Loading Debug ===")
    
    model_files = [
        "models/alphabet_model.keras",
        "models/word_model.keras",
        "models/alphabet_label_map.json",
        "models/word_label_map.json"
    ]
    
    for file_path in model_files:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"  {file_path}: {'✅' if exists else '❌'} ({size} bytes)")
    
    try:
        # Test loading models
        alphabet_model = tf.keras.models.load_model("models/alphabet_model.keras")
        word_model = tf.keras.models.load_model("models/word_model.keras")
        
        print(f"  Alphabet model input shape: {alphabet_model.input_shape}")
        print(f"  Word model input shape: {word_model.input_shape}")
        
        # Test label maps
        with open("models/alphabet_label_map.json", 'r') as f:
            alphabet_map = json.load(f)
            print(f"  Alphabet classes: {len(alphabet_map)}")
            
        with open("models/word_label_map.json", 'r') as f:
            word_map = json.load(f)
            print(f"  Word classes: {len(word_map)}")
            
    except Exception as e:
        print(f"  ❌ Error loading models: {e}")

def debug_preprocessing():
    """Debug image preprocessing"""
    print("\n=== Preprocessing Debug ===")
    
    # Create test images
    test_images = {
        "black": np.zeros((100, 100, 3), dtype=np.uint8),
        "white": np.ones((100, 100, 3), dtype=np.uint8) * 255,
        "random": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    }
    
    for name, img in test_images.items():
        try:
            processed = preprocess_image(img)
            if processed is not None:
                print(f"  {name}: shape={processed.shape}, min={processed.min():.3f}, max={processed.max():.3f}")
            else:
                print(f"  {name}: preprocessing failed")
        except Exception as e:
            print(f"  {name}: error - {e}")

def debug_prediction_pipeline():
    """Test prediction pipeline with dummy data"""
    print("\n=== Prediction Pipeline Debug ===")
    
    # Create dummy test images
    dummy_alphabet = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    dummy_word_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    
    try:
        # Test alphabet prediction
        alphabet_result = predict_alphabet(dummy_alphabet)
        print(f"  Alphabet prediction: {alphabet_result}")
        
        # Test word prediction
        word_result = predict_word_sequence(dummy_word_frames)
        print(f"  Word prediction: {word_result}")
        
    except Exception as e:
        print(f"  ❌ Prediction error: {e}")

def debug_camera_capture():
    """Debug camera capture"""
    print("\n=== Camera Capture Debug ===")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera working: frame shape={frame.shape}, dtype={frame.dtype}")
                print(f"  Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
            else:
                print("  ❌ Cannot read frame from camera")
        else:
            print("  ❌ Cannot open camera")
        cap.release()
    except Exception as e:
        print(f"  ❌ Camera error: {e}")

def debug_label_mapping():
    """Debug label mapping"""
    print("\n=== Label Mapping Debug ===")
    
    try:
        # Test alphabet label mapping
        with open("models/alphabet_label_map.json", 'r') as f:
            alphabet_map = json.load(f)
            reverse_map = {v: k for k, v in alphabet_map.items()}
            
        print(f"  Alphabet labels: {len(alphabet_map)}")
        print(f"  Sample mappings: {dict(list(reverse_map.items())[:5])}")
        
        # Test word label mapping
        with open("models/word_label_map.json", 'r') as f:
            word_map = json.load(f)
            reverse_map = {v: k for k, v in word_map.items()}
            
        print(f"  Word labels: {len(word_map)}")
        print(f"  Sample mappings: {dict(list(reverse_map.items())[:5])}")
        
    except Exception as e:
        print(f"  ❌ Label mapping error: {e}")

def run_comprehensive_debug():
    """Run all debugging tests"""
    print("🔍 Kannada Sign Language Prediction Debug")
    print("=" * 50)
    
    debug_model_loading()
    debug_preprocessing()
    debug_prediction_pipeline()
    debug_camera_capture()
    debug_label_mapping()
    
    print("\n✅ Debug complete. Check output above for issues.")

if __name__ == "__main__":
    run_comprehensive_debug()

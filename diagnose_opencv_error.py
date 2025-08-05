#!/usr/bin/env python3
"""
Diagnostic script to identify the exact cause of OpenCV C++ exception
"""

import cv2
import numpy as np
import os
import sys
import traceback
from scripts.predict_updated import load_models, predict_alphabet

def diagnose_camera():
    """Test camera access and identify issues"""
    print("🔍 Diagnosing camera access...")
    
    # Test camera indices
    for i in range(5):
        print(f"\nTesting camera index {i}:")
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  ✅ Camera {i} accessible")
                
                # Test frame reading
                ret, frame = cap.read()
                if ret:
                    print(f"  ✅ Frame read successful - Shape: {frame.shape}")
                    print(f"  ✅ Frame dtype: {frame.dtype}")
                    print(f"  ✅ Frame min/max values: {frame.min()}/{frame.max()}")
                else:
                    print(f"  ❌ Failed to read frame")
                
                cap.release()
            else:
                print(f"  ❌ Camera {i} not accessible")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            traceback.print_exc()

def diagnose_image_processing():
    """Test image preprocessing pipeline"""
    print("\n🔍 Diagnosing image processing...")
    
    # Test with sample images
    test_images = [
        "data/alphabet_images/ಕ/1.jpg",
        "data/alphabet_images/ಖ/1.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting {img_path}:")
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  ❌ Failed to load image")
                    continue
                
                print(f"  ✅ Loaded - Shape: {image.shape}, Dtype: {image.dtype}")
                
                # Test preprocessing
                from scripts.predict_updated import preprocess_image
                features = preprocess_image(image)
                print(f"  ✅ Preprocessing successful - Features: {len(features)}")
                
                # Test prediction
                result = predict_alphabet(image)
                print(f"  ✅ Prediction: {result}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                traceback.print_exc()

def diagnose_model_loading():
    """Test model loading and validation"""
    print("\n🔍 Diagnosing model loading...")
    
    try:
        # Test model loading
        success = load_models()
        print(f"  Model loading: {'✅ Success' if success else '❌ Failed'}")
        
        # Check file existence
        model_files = [
            "models/alphabet_model_retrained.pkl",
            "models/alphabet_scaler_retrained.pkl",
            "models/alphabet_label_map_retrained.json"
        ]
        
        for file_path in model_files:
            exists = os.path.exists(file_path)
            print(f"  {file_path}: {'✅ Exists' if exists else '❌ Missing'}")
            
    except Exception as e:
        print(f"  ❌ Model loading error: {e}")
        traceback.print_exc()

def diagnose_system_resources():
    """Check system resources and OpenCV version"""
    print("\n🔍 Diagnosing system resources...")
    
    try:
        # OpenCV version
        print(f"  OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV operations
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = cv2.resize(test_img, (50, 50))
        print(f"  ✅ Basic OpenCV operations working")
        
        # Test color space conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print(f"  ✅ Color space conversion working")
        
    except Exception as e:
        print(f"  ❌ OpenCV operations failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 OpenCV Error Diagnostic Tool")
    print("=" * 50)
    
    diagnose_camera()
    diagnose_image_processing()
    diagnose_model_loading()
    diagnose_system_resources()
    
    print("\n✅ Diagnostic complete! Check the output above for specific issues.")

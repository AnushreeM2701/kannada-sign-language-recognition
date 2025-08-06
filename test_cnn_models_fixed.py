#!/usr/bin/env python3
"""
Fixed test script to verify CNN models are working correctly after training
"""

import cv2
import numpy as np
import os
from scripts.cnn_predict_fixed import predict_alphabet, predict_word_sequence

def test_cnn_models():
    """Test the newly trained CNN models with fixed input shapes"""
    print("🧪 Testing CNN Models After Training (Fixed)")
    print("=" * 50)
    
    # Create test images
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test 1: Alphabet model
    print("\n📊 Testing Alphabet CNN Model...")
    try:
        result = predict_alphabet(test_image)
        if result:
            print(f"✅ Alphabet prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
        else:
            print("❌ Alphabet prediction failed")
    except Exception as e:
        print(f"❌ Alphabet model error: {e}")
    
    # Test 2: Word model with correct sequence length
    print("\n📊 Testing Word CNN Model...")
    try:
        # Create exactly 10 frames as expected by the model
        frames = [test_image] * 10
        result = predict_word_sequence(frames)
        if result:
            print(f"✅ Word prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
        else:
            print("❌ Word prediction failed")
    except Exception as e:
        print(f"❌ Word model error: {e}")
    
    print("\n🎉 CNN Model Testing Complete!")

if __name__ == "__main__":
    test_cnn_models()

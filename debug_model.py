#!/usr/bin/env python3
"""
Debug script to test the model with sample images from the data folder
This will help identify why all predictions are showing "ಛ"
"""

import cv2
import numpy as np
import os
import json
from scripts.predict_updated import predict_alphabet

def load_sample_images():
    """Load sample images from the alphabet dataset for testing"""
    data_dir = "data/alphabet_images"
    samples = []
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found")
        return samples
    
    # Get first few images from each alphabet folder
    alphabet_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    for alphabet in alphabet_folders[:5]:  # Test first 5 alphabets
        alphabet_path = os.path.join(data_dir, alphabet)
        images = [f for f in os.listdir(alphabet_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            # Take first 3 images from each alphabet
            for img_name in images[:3]:
                img_path = os.path.join(alphabet_path, img_name)
                samples.append({
                    'path': img_path,
                    'expected': alphabet,
                    'filename': img_name
                })
    
    return samples

def test_model_accuracy():
    """Test model accuracy with known samples"""
    print("🔍 Testing model with sample images...")
    print("=" * 50)
    
    samples = load_sample_images()
    
    if not samples:
        print("❌ No samples found to test")
        return
    
    correct = 0
    total = len(samples)
    
    for sample in samples:
        try:
            # Load and test image
            image = cv2.imread(sample['path'])
            if image is None:
                print(f"❌ Could not load: {sample['path']}")
                continue
            
            # Convert BGR to RGB (since predict_alphabet expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            predicted = predict_alphabet(image_rgb)
            
            # Check if correct
            is_correct = predicted == sample['expected']
            status = "✅" if is_correct else "❌"
            
            print(f"{status} {sample['expected']} -> {predicted} | {sample['filename']}")
            
            if is_correct:
                correct += 1
                
        except Exception as e:
            print(f"❌ Error processing {sample['path']}: {e}")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print("=" * 50)
    print(f"📊 Model Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    
    if accuracy < 50:
        print("⚠️  Low accuracy detected - model may need retraining")
    elif accuracy >= 80:
        print("✅ Model appears to be working correctly")
    else:
        print("⚠️  Moderate accuracy - check data quality and preprocessing")

def test_single_image(image_path):
    """Test a single image and show detailed prediction info"""
    print(f"\n🔍 Testing single image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Show image info
        print(f"📸 Image shape: {image.shape}")
        print(f"🎨 Color space: RGB")
        
        # Get prediction
        predicted = predict_alphabet(image_rgb)
        print(f"🎯 Predicted: {predicted}")
        
        # Show a preview of the image
        cv2.imshow("Test Image", image)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Model Debug Script")
    print("=" * 50)
    
    # Test model accuracy with samples
    test_model_accuracy()
    
    # Test a specific image if provided
    test_image = "data/alphabet_images/ಕ/1.jpg"  # Test with a known image
    if os.path.exists(test_image):
        test_single_image(test_image)
    else:
        print(f"\n⚠️  Sample test image not found: {test_image}")
    
    print("\n✅ Debug testing complete!")

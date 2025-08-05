#!/usr/bin/env python3
"""
Fix script for OpenCV C++ exceptions in sign language prediction
Addresses camera access, image preprocessing, and model loading issues
"""

import cv2
import numpy as np
import os
import sys
import json
import joblib
from pathlib import Path

class OpenCVFixer:
    def __init__(self):
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
    def fix_camera_access(self):
        """Fix camera access issues with proper error handling"""
        print("🔧 Fixing camera access...")
        
        # Try multiple camera indices
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_ANY)  # Use any available backend
                
                if cap.isOpened():
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test frame reading
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Camera {i} working properly")
                        self.camera_index = i
                        cap.release()
                        return True
                    else:
                        print(f"❌ Camera {i} accessible but frame read failed")
                        
                cap.release()
                
            except Exception as e:
                print(f"❌ Camera {i} error: {e}")
                
        return False
    
    def fix_image_preprocessing(self, image):
        """Fix image preprocessing issues with robust error handling"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
            
        try:
            # Ensure image is in correct format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
            # Ensure valid dimensions
            height, width = image.shape[:2]
            if height == 0 or width == 0:
                raise ValueError("Image has zero dimensions")
                
            # Resize with proper interpolation
            if height != 64 or width != 64:
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize safely
            image_normalized = image_rgb.astype(np.float32) / 255.0
            
            # Ensure no NaN or inf values
            if np.any(np.isnan(image_normalized)) or np.any(np.isinf(image_normalized)):
                image_normalized = np.nan_to_num(image_normalized, nan=0.0, posinf=1.0, neginf=0.0)
                
            return image_normalized.flatten()
            
        except cv2.error as e:
            raise RuntimeError(f"OpenCV preprocessing error: {e}")
        except Exception as e:
            raise RuntimeError(f"Preprocessing error: {e}")
    
    def fix_model_loading(self):
        """Fix model loading issues with fallback options"""
        print("🔧 Fixing model loading...")
        
        # Define fallback model configurations
        model_configs = [
            {
                'model': 'models/alphabet_model_retrained.pkl',
                'scaler': 'models/alphabet_scaler_retrained.pkl',
                'labels': 'models/alphabet_label_map_retrained.json',
                'name': 'retrained'
            },
            {
                'model': 'models/alphabet_model_lightweight.pkl',
                'scaler': 'models/alphabet_scaler.pkl',
                'labels': 'utils/label_map.json',
                'name': 'lightweight'
            },
            {
                'model': 'models/alphabet_model.keras',
                'scaler': 'models/alphabet_scaler.pkl',
                'labels': 'utils/label_map.json',
                'name': 'keras'
            }
        ]
        
        for config in model_configs:
            try:
                model_path = config['model']
                scaler_path = config['scaler']
                labels_path = config['labels']
                
                # Check if files exist
                if not all(os.path.exists(f) for f in [model_path, scaler_path, labels_path]):
                    print(f"  ⚠️  Missing {config['name']} files")
                    continue
                
                # Load models
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Load labels
                with open(labels_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                    labels = {v: k for k, v in label_map.items()}
                
                print(f"✅ Successfully loaded {config['name']} model")
                return model, scaler, labels
                
            except Exception as e:
                print(f"  ❌ Failed to load {config['name']}: {e}")
                continue
        
        raise RuntimeError("All model loading attempts failed")
    
    def create_safe_prediction_function(self):
        """Create a safe prediction function with all fixes applied"""
        print("🔧 Creating safe prediction function...")
        
        # Fix camera access
        if not self.fix_camera_access():
            raise RuntimeError("Camera access could not be fixed")
        
        # Fix model loading
        model, scaler, labels = self.fix_model_loading()
        
        def safe_predict(image):
            """Safe prediction function with all fixes applied"""
            try:
                if image is None:
                    return "Error: No image provided"
                
                # Apply image preprocessing fixes
                features = self.fix_image_preprocessing(image)
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Make prediction
                prediction_idx = int(model.predict(features_scaled)[0])
                
                # Get label
                if prediction_idx in labels:
                    return labels[prediction_idx]
                else:
                    return f"Error: Invalid prediction index {prediction_idx}"
                    
            except Exception as e:
                return f"Error: {str(e)}"
        
        return safe_predict

def main():
    """Main fix function"""
    print("🚀 OpenCV Exception Fix Tool")
    print("=" * 50)
    
    try:
        fixer = OpenCVFixer()
        
        # Apply all fixes
        safe_predict = fixer.create_safe_prediction_function()
        
        print("\n✅ All fixes applied successfully!")
        print("You can now use the safe_predict() function for predictions")
        
        # Test with a sample
        test_image = np.zeros((100, 100, 3), dtype=np.uint8) + 255
        result = safe_predict(test_image)
        print(f"Test prediction: {result}")
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

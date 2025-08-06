#!/usr/bin/env python3
"""
Updated prediction script using CNN models
Handles image preprocessing for CNN models
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os

# Global variables for CNN models
alphabet_model = None
alphabet_labels = None
word_model = None
word_labels = None

def load_models():
    """Load CNN models with comprehensive error handling"""
    global alphabet_model, alphabet_labels, word_model, word_labels
    
    try:
        # Define model paths
        model_configs = [
            {
                'model': 'models/alphabet_model.keras',
                'labels': 'models/alphabet_label_map.json',
                'name': 'alphabet'
            },
            {
                'model': 'models/word_model.keras',
                'labels': 'models/word_label_map.json',
                'name': 'word'
            }
        ]
        
        for config in model_configs:
            try:
                model_path = config['model']
                label_map_path = config['labels']
                
                # Check if files exist
                if not os.path.exists(model_path):
                    print(f"⚠️  Missing {config['name']} model: {model_path}")
                    continue
                
                # Load CNN model
                print(f"📂 Loading {config['name']} CNN model...")
                if config['name'] == 'alphabet':
                    alphabet_model = tf.keras.models.load_model(model_path)
                else:
                    word_model = tf.keras.models.load_model(model_path)
                
                # Load label mapping
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                    if config['name'] == 'alphabet':
                        alphabet_labels = {v: k for k, v in label_map.items()}
                    else:
                        word_labels = {v: k for k, v in label_map.items()}
                
                print(f"✅ Successfully loaded {config['name']} CNN model with {len(label_map)} classes")
                
            except Exception as e:
                print(f"❌ Failed to load {config['name']} CNN model: {e}")
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ Critical error loading CNN models: {e}")
        return False

def preprocess_image(image, target_size=64):
    """Consistent preprocessing for prediction with enhanced error handling"""
    try:
        if image is None:
            raise ValueError("Image is None")
        
        # Validate image shape and type
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if image.size == 0:
            raise ValueError("Image is empty")
        
        # Get image dimensions
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Image has zero dimensions")
        
        # Ensure we have a valid image
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # Handle different image formats safely
        if channels == 3:
            # BGR to RGB conversion
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            # BGRA to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif channels == 1:
            # Grayscale to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image format: {channels} channels")
        
        # Validate resize operation
        if target_size <= 0:
            raise ValueError("Target size must be positive")
        
        # Resize with interpolation
        img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Validate resize result
        if img_resized.shape != (target_size, target_size, 3):
            raise ValueError("Resize operation failed")
        
        # Normalize pixel values safely
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Ensure values are in valid range
        if np.any(img_normalized < 0) or np.any(img_normalized > 1):
            raise ValueError("Normalization produced invalid values")
        
        # Flatten for sklearn
        features = img_normalized.flatten()
        
        # Validate feature vector
        expected_features = target_size * target_size * 3
        if len(features) != expected_features:
            raise ValueError(f"Feature vector has wrong size: {len(features)} != {expected_features}")
        
        return features
        
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error during preprocessing: {e}")
    except Exception as e:
        raise RuntimeError(f"Preprocessing error: {e}")

def predict_alphabet(image):
    """Predict alphabet from image with enhanced error handling"""
    try:
        # Validate models are loaded
        if alphabet_model is None or alphabet_scaler is None:
            if not load_models():
                raise RuntimeError("Failed to load models")
        
        # Validate inputs
        if alphabet_labels is None:
            raise RuntimeError("Label mapping not loaded")
        
        # Preprocess image with error handling
        features = preprocess_image(image)
        
        # Validate feature vector
        if features is None or len(features) == 0:
            raise ValueError("Failed to extract features from image")
        
        # Scale features safely
        try:
            features_scaled = alphabet_scaler.transform([features])
        except Exception as e:
            raise RuntimeError(f"Feature scaling failed: {e}")
        
        # Make prediction
        try:
            prediction_idx = int(alphabet_model.predict(features_scaled)[0])
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        
        # Validate prediction index
        if prediction_idx not in alphabet_labels:
            raise ValueError(f"Invalid prediction index: {prediction_idx}")
        
        # Get label
        predicted_label = alphabet_labels[prediction_idx]
        
        # Validate label
        if predicted_label is None or predicted_label == "":
            raise ValueError("Empty prediction label")
        
        return predicted_label
        
    except cv2.error as e:
        print(f"❌ OpenCV error in prediction: {e}")
        return f"OpenCV Error: {str(e)}"
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error: {str(e)}"

def predict_alphabet_with_confidence(image):
    """Predict alphabet with confidence scores"""
    if alphabet_model is None or alphabet_scaler is None:
        if not load_models():
            return "Error", {}
    
    try:
        features = preprocess_image(image)
        features_scaled = alphabet_scaler.transform([features])
        
        # Get prediction probabilities
        if hasattr(alphabet_model, 'predict_proba'):
            probabilities = alphabet_model.predict_proba(features_scaled)[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            predicted_label = alphabet_labels[prediction_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = {
                alphabet_labels[idx]: float(probabilities[idx]) 
                for idx in top_indices
            }
            
            return predicted_label, top_predictions
        else:
            prediction_idx = alphabet_model.predict(features_scaled)[0]
            predicted_label = alphabet_labels[prediction_idx]
            return predicted_label, {}
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Error", {}

def test_model_quick():
    """Quick test of the model with sample images"""
    print("🧪 Quick model test...")
    
    # Test with a few sample images
    test_images = [
        "data/alphabet_images/ಕ/1.jpg",
        "data/alphabet_images/ಖ/1.jpg",
        "data/alphabet_images/ಗ/1.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            try:
                image = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                predicted, confidences = predict_alphabet_with_confidence(image_rgb)
                expected = os.path.basename(os.path.dirname(img_path))
                
                status = "✅" if predicted == expected else "❌"
                print(f"{status} Expected: {expected}, Predicted: {predicted}")
                
                if confidences:
                    print(f"   Top predictions: {dict(list(confidences.items())[:3])}")
                    
            except Exception as e:
                print(f"❌ Error testing {img_path}: {e}")

# Initialize models on import
load_models()

# Global variables for word model
word_model = None
word_scaler = None
word_labels = None

def load_word_model():
    """Load the word recognition model and related artifacts"""
    global word_model, word_scaler, word_labels
    try:
        if word_model is not None and word_scaler is not None and word_labels is not None:
            return True
        
        import joblib
        import json
        
        model_path = "models/word_model_lightweight.pkl"
        scaler_path = "models/word_scaler.pkl"
        label_map_path = "utils/word_label_map.json"
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(label_map_path)):
            print("⚠️ Missing word model files")
            return False
        
        word_model = joblib.load(model_path)
        word_scaler = joblib.load(scaler_path)
        
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
            word_labels = {v: k for k, v in label_map.items()}
        
        print(f"✅ Successfully loaded word model with {len(word_labels)} classes")
        return True
    except Exception as e:
        print(f"❌ Failed to load word model: {e}")
        return False

def extract_features_from_frames(frames, target_frames=10):
    """Extract features from a list of frames for word prediction"""
    import numpy as np
    import cv2
    
    feature_vector = []
    for frame in frames:
        resized = cv2.resize(frame, (64, 16))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        feature_vector.extend(gray.flatten())
    
    for frame in frames:
        hist_b = cv2.calcHist([frame[:,:,0]], [0], None, [8], [0, 256]).flatten()
        hist_g = cv2.calcHist([frame[:,:,1]], [0], None, [8], [0, 256]).flatten()
        hist_r = cv2.calcHist([frame[:,:,2]], [0], None, [8], [0, 256]).flatten()
        feature_vector.extend(np.concatenate([hist_b, hist_g, hist_r]))
    
    return np.array(feature_vector)

def predict_word_sequence(frames):
    """Predict the word label from a sequence of frames"""
    if not load_word_model():
        return "Error: Word model not loaded"
    
    try:
        features = extract_features_from_frames(frames)
        features_scaled = word_scaler.transform([features])
        prediction_idx = int(word_model.predict(features_scaled)[0])
        
        if prediction_idx not in word_labels:
            return "Unknown"
        
        predicted_label = word_labels[prediction_idx]
        return predicted_label
    except Exception as e:
        print(f"❌ Word prediction error: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    test_model_quick()

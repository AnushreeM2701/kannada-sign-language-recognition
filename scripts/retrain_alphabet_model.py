#!/usr/bin/env python3
"""
Complete retraining script for Kannada alphabet model
Addresses model corruption and accuracy issues
"""

import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data/alphabet_images"
MODEL_DIR = "models"
IMG_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42

def validate_data():
    """Validate data structure and count images per alphabet"""
    print("🔍 Validating data structure...")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} not found")
    
    alphabet_counts = {}
    total_images = 0
    
    for alphabet in sorted(os.listdir(DATA_DIR)):
        alphabet_path = os.path.join(DATA_DIR, alphabet)
        if not os.path.isdir(alphabet_path):
            continue
            
        images = [f for f in os.listdir(alphabet_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        count = len(images)
        alphabet_counts[alphabet] = count
        total_images += count
        
        if count < 5:
            print(f"⚠️  {alphabet}: Only {count} images (minimum 5 recommended)")
    
    print(f"✅ Found {len(alphabet_counts)} alphabets with {total_images} total images")
    return alphabet_counts

def load_and_preprocess_data():
    """Load and preprocess all images with consistent feature extraction"""
    print("📸 Loading and preprocessing images...")
    
    images = []
    labels = []
    label_map = {}
    label_idx = 0
    
    alphabet_counts = validate_data()
    
    for alphabet in sorted(alphabet_counts.keys()):
        if alphabet_counts[alphabet] < 5:
            continue
            
        label_map[alphabet] = label_idx
        alphabet_path = os.path.join(DATA_DIR, alphabet)
        
        images_in_folder = [f for f in os.listdir(alphabet_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images_in_folder:
            img_path = os.path.join(alphabet_path, img_name)
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to RGB and resize
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                
                # Normalize pixel values
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Flatten for sklearn
                features = img_normalized.flatten()
                
                images.append(features)
                labels.append(label_idx)
                
            except Exception as e:
                print(f"⚠️  Error processing {img_path}: {e}")
                continue
        
        label_idx += 1
    
    if len(images) == 0:
        raise ValueError("No valid images found!")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Loaded {len(X)} samples across {len(label_map)} classes")
    return X, y, label_map

def train_robust_model(X, y, label_map):
    """Train a robust Random Forest model with proper validation"""
    print("🧠 Training robust model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Further split training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_accuracy = model.score(X_val_scaled, y_val)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"📊 Validation accuracy: {val_accuracy:.4f}")
    print(f"📊 Test accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=list(label_map.keys())))
    
    return model, scaler, test_accuracy

def save_models(model, scaler, label_map, accuracy):
    """Save the trained model and supporting files"""
    print("💾 Saving models...")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, "alphabet_model_retrained.pkl")
    scaler_path = os.path.join(MODEL_DIR, "alphabet_scaler_retrained.pkl")
    label_map_path = os.path.join(MODEL_DIR, "alphabet_label_map_retrained.json")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Label map saved: {label_map_path}")
    
    # Create a summary file
    summary = {
        "accuracy": float(accuracy),
        "num_classes": len(label_map),
        "image_size": IMG_SIZE,
        "model_type": "RandomForestClassifier",
        "training_date": str(np.datetime64('today'))
    }
    
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return model_path, scaler_path, label_map_path

def main():
    """Main training pipeline"""
    print("🚀 Kannada Alphabet Model Retraining")
    print("=" * 50)
    
    try:
        # Validate and load data
        X, y, label_map = load_and_preprocess_data()
        
        # Train model
        model, scaler, accuracy = train_robust_model(X, y, label_map)
        
        # Save everything
        model_path, scaler_path, label_map_path = save_models(model, scaler, label_map, accuracy)
        
        print("\n🎉 Retraining complete!")
        print(f"   Model accuracy: {accuracy:.2%}")
        print(f"   Classes: {len(label_map)}")
        print(f"   Model files saved in: {MODEL_DIR}")
        
        # Test the new model
        print("\n🧪 Testing new model...")
        test_features = X[0:5]  # Test with first 5 samples
        test_scaled = scaler.transform(test_features)
        predictions = model.predict(test_scaled)
        
        for i, pred in enumerate(predictions[:3]):
            predicted_label = list(label_map.keys())[list(label_map.values()).index(pred)]
            print(f"   Sample {i+1}: Predicted as {predicted_label}")
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        raise

if __name__ == "__main__":
    main()

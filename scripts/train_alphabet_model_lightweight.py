import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Paths
DATA_DIR = "data/alphabet_images"
MODEL_PATH = "models/alphabet_model_lightweight.pkl"
LABEL_MAP_PATH = "utils/label_map.json"
SCALER_PATH = "models/alphabet_scaler.pkl"

# Parameters
IMG_SIZE = 64

def load_images_and_labels(data_dir, img_size):
    """
    Loads images and their labels from the specified directory.
    Returns:
        images: np.ndarray of shape (num_samples, img_size, img_size, 3)
        labels: np.ndarray of shape (num_samples,)
        label_map: dict mapping label names to integer indices
    """
    images = []
    labels = []
    label_map = {}
    idx = 0

    print("[INFO] Loading images...")
    for label_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder_path) or label_name.startswith('.'):
            continue  # Skip files like .DS_Store or hidden folders

        image_files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
        if len(image_files) == 0:
            continue  # Skip empty folders

        label_map[label_name] = idx

        for file in image_files:
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] Failed to load: {img_path}")
                    continue
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"[WARNING] Skipping: {img_path} due to error: {e}")
        idx += 1

    if len(images) == 0:
        raise RuntimeError("No images loaded. Please check your data directory.")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), label_map

def extract_features(images):
    """
    Extract features from images for traditional ML approach.
    """
    print("[INFO] Extracting features...")
    features = []
    
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to smaller size for feature extraction
        small = cv2.resize(gray, (32, 32))
        
        # Flatten the image
        flat = small.flatten()
        
        # Add color histogram features
        hist_b = cv2.calcHist([img[:,:,0]], [0], None, [16], [0, 256]).flatten()
        hist_g = cv2.calcHist([img[:,:,1]], [0], None, [16], [0, 256]).flatten()
        hist_r = cv2.calcHist([img[:,:,2]], [0], None, [16], [0, 256]).flatten()
        
        # Combine all features
        combined = np.concatenate([flat, hist_b, hist_g, hist_r])
        features.append(combined)
    
    return np.array(features)

def main():
    # Load data
    images, labels, label_map = load_images_and_labels(DATA_DIR, IMG_SIZE)
    images = images / 255.0  # Normalize
    
    # Extract features
    features = extract_features(images)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("[INFO] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[INFO] Test accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))
    
    # Save model, scaler, and label map
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Lightweight model saved to {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved to {SCALER_PATH}")
    
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Label map saved to {LABEL_MAP_PATH}")
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print("\n[INFO] Top 10 most important features:", indices)

if __name__ == "__main__":
    main()

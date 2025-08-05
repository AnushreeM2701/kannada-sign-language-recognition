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
DATA_DIR = "data/word_videos"
MODEL_PATH = "models/word_model_lightweight.pkl"
LABEL_MAP_PATH = "utils/word_label_map.json"
SCALER_PATH = "models/word_scaler.pkl"

# Parameters
IMG_SIZE = 128  # Different size for word recognition
FRAME_HEIGHT = 32

def extract_frames_from_video(video_path, target_frames=10):
    """
    Extract frames from a video file.
    Returns a list of frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return frames
    
    # Calculate frame indices to extract
    if total_frames <= target_frames:
        indices = range(total_frames)
    else:
        step = total_frames // target_frames
        indices = range(0, total_frames, step)[:target_frames]
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (IMG_SIZE, FRAME_HEIGHT))
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim to ensure consistent number of frames
    while len(frames) < target_frames:
        frames.append(np.zeros((FRAME_HEIGHT, IMG_SIZE, 3), dtype=np.uint8))
    
    return frames[:target_frames]

def load_videos_and_labels(data_dir, target_frames=10):
    """
    Load videos and their labels.
    Returns:
        features: np.ndarray of extracted features
        labels: np.ndarray of labels
        label_map: dict mapping label names to indices
    """
    features = []
    labels = []
    label_map = {}
    idx = 0
    
    print("[INFO] Loading videos...")
    
    for label_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder_path) or label_name.startswith('.'):
            continue
        
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if len(video_files) == 0:
            continue
        
        label_map[label_name] = idx
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            try:
                frames = extract_frames_from_video(video_path, target_frames)
                if len(frames) == target_frames:
                    # Flatten all frames into a single feature vector
                    feature_vector = []
                    for frame in frames:
                        # Resize and flatten each frame
                        resized = cv2.resize(frame, (64, 16))  # Smaller size for efficiency
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        feature_vector.extend(gray.flatten())
                    
                    # Add color histogram features
                    for frame in frames:
                        hist_b = cv2.calcHist([frame[:,:,0]], [0], None, [8], [0, 256]).flatten()
                        hist_g = cv2.calcHist([frame[:,:,1]], [0], None, [8], [0, 256]).flatten()
                        hist_r = cv2.calcHist([frame[:,:,2]], [0], None, [8], [0, 256]).flatten()
                        feature_vector.extend(np.concatenate([hist_b, hist_g, hist_r]))
                    
                    features.append(feature_vector)
                    labels.append(idx)
                    
            except Exception as e:
                print(f"[WARNING] Skipping {video_path}: {e}")
        
        idx += 1
    
    if len(features) == 0:
        raise RuntimeError("No videos loaded. Please check your data directory.")
    
    return np.array(features), np.array(labels), label_map

def main():
    # Load data
    features, labels, label_map = load_videos_and_labels(DATA_DIR)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("[INFO] Training word recognition model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
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
    print(f"[INFO] Word model saved to {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved to {SCALER_PATH}")
    
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Word label map saved to {LABEL_MAP_PATH}")
    
    return model, scaler, label_map

if __name__ == "__main__":
    main()

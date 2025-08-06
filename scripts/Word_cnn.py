import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Config
DATA_DIR = "data/word_videos"
MODEL_DIR = "models"
IMG_SIZE = 64
NUM_FRAMES = 10
BATCH_SIZE = 8
EPOCHS = 20

def extract_video_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype('float32') / 255.0
        frames.append(frame)
        if len(frames) == num_frames:
            break

    cap.release()

    # Pad if fewer frames
    while len(frames) < num_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames[:num_frames])

def load_dataset():
    X, y = [], []
    label_map = {}
    label_index = 0

    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        label_map[label] = label_index

        for file in os.listdir(label_path):
            if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(label_path, file)
            try:
                frames = extract_video_frames(video_path)
                X.append(frames)
                y.append(label_index)
            except Exception as e:
                print(f"❌ Error reading {video_path}: {e}")

        label_index += 1

    return np.array(X), np.array(y), label_map

def build_model(input_shape, num_classes):
    model = Sequential()

    # TimeDistributed CNN
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    # LSTM for sequence modeling
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    print("🚀 Loading video data...")
    X, y, label_map = load_dataset()
    print(f"✅ Loaded {len(X)} samples from {len(label_map)} classes.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Build model
    input_shape = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    model = build_model(input_shape, num_classes=len(label_map))

    # Train
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "word_model.keras"))

    # Save label map
    with open(os.path.join(MODEL_DIR, "word_label_map.json"), "w", encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Save summary
    with open(os.path.join(MODEL_DIR, "word_model_summary.json"), "w") as f:
        json.dump({
            "date": str(datetime.now()),
            "classes": len(label_map),
            "input_shape": input_shape,
            "epochs": EPOCHS
        }, f, indent=2)

    print("🎉 Training complete! Model and label map saved.")

if __name__ == "__main__":
    main()

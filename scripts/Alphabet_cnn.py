#!/usr/bin/env python3
"""
CNN-based Kannada Alphabet Recognition Model Training
"""

import os
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Configuration
DATA_DIR = "data/alphabet_images"
MODEL_DIR = "models"
IMG_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_STATE = 42

# Step 1: Load and preprocess data
def load_dataset():
    images = []
    labels = []
    
    print("📥 Loading images...")
    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        
        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                images.append(img)
                labels.append(label)
    
    print(f"✅ Loaded {len(images)} images.")
    return np.array(images), np.array(labels)

# Step 2: Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Step 3: Save model and metadata
def save_model(model, label_map):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "alphabet_model.keras")
    model.save(model_path)

    with open(os.path.join(MODEL_DIR, "alphabet_label_map.json"), "w", encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Model saved at {model_path}")
    print(f"✅ Label map saved.")

# Main training pipeline
def main():
    print("🚀 Training CNN-based Kannada Alphabet Model")
    print("=" * 50)
    
    X, y = load_dataset()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Save label map
    label_map = {label: int(idx) for idx, label in enumerate(le.classes_)}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Build and train model
    model = build_model((IMG_SIZE, IMG_SIZE, 3), num_classes=y_categorical.shape[1])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\n📊 Test Accuracy: {test_accuracy:.2%}")
    
    # Save model
    save_model(model, label_map)
    
    # Summary
    with open(os.path.join(MODEL_DIR, "training_summary.json"), "w") as f:
        json.dump({
            "model_type": "CNN",
            "input_shape": (IMG_SIZE, IMG_SIZE, 3),
            "epochs": EPOCHS,
            "test_accuracy": float(test_accuracy),
            "trained_on": str(datetime.now())
        }, f, indent=2)
    
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()

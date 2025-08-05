import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# For alphabet image dataset
def load_alphabet_dataset(data_dir, img_size=(64, 64)):
    X, y = [], []
    class_labels = sorted(os.listdir(data_dir))
    label_map = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label_map[label])

    X = np.array(X) / 255.0  # normalize
    y = to_categorical(y, num_classes=len(class_labels))
    return train_test_split(X, y, test_size=0.2), label_map

# For word video dataset
def extract_frames_from_video(video_path, frame_count=20, img_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // frame_count)

    count = 0
    while len(frames) < frame_count and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
        count += 1

    cap.release()
    frames = np.array(frames)
    return frames

def load_word_video_dataset(video_dir, frame_count=20, img_size=(64, 64)):
    X, y = [], []
    class_labels = sorted(os.listdir(video_dir))
    label_map = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        label_path = os.path.join(video_dir, label)
        if not os.path.isdir(label_path):
            continue
        for video_name in os.listdir(label_path):
            video_path = os.path.join(label_path, video_name)
            frames = extract_frames_from_video(video_path, frame_count, img_size)
            if len(frames) == frame_count:
                X.append(frames)
                y.append(label_map[label])

    X = np.array(X) / 255.0  # normalize
    y = to_categorical(y, num_classes=len(class_labels))
    return train_test_split(X, y, test_size=0.2), label_map

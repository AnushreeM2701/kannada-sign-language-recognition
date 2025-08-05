
import cv2
import numpy as np

def clean_alphabet_image(img, to_rgb=True):
    """
    Preprocess an alphabet image for model input.
    - Resizes to (64, 64)
    - Normalizes pixel values to [0, 1]
    - Expands dimensions to (1, 64, 64, 3)
    - Optionally converts BGR to RGB
    """
    if img is None:
        raise ValueError("Input image is None")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (64, 64))
    normalized = resized / 255.0
    processed = np.expand_dims(normalized, axis=0)
    return processed

def clean_word_image(img, to_rgb=True):
    """
    Preprocess a word image for model input.
    - Resizes to (128, 32)
    - Normalizes pixel values to [0, 1]
    - Expands dimensions to (1, 32, 128, 3)
    - Optionally converts BGR to RGB
    """
    if img is None:
        raise ValueError("Input image is None")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (128, 32))
    normalized = resized / 255.0
    processed = np.expand_dims(normalized, axis=0)
    return processed
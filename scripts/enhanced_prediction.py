import tensorflow as tf
import numpy as np
import cv2
import json
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

class EnhancedPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.alphabet_model = None
        self.word_model = None
        self.alphabet_labels = {}
        self.word_labels = {}
        self.alphabet_img_size = (224, 224)
        self.word_img_size = (112, 112)
        self.word_max_frames = 30
        
        self.load_models()
    
    def load_models(self):
        """Load both alphabet and word models"""
        try:
            # Load alphabet model
            alphabet_path = os.path.join(self.model_dir, 'enhanced_alphabet_model.keras')
            if os.path.exists(alphabet_path):
                self.alphabet_model = tf.keras.models.load_model(alphabet_path)
                print("✅ Enhanced alphabet model loaded")
            
            # Load word model
            word_path = os.path.join(self.model_dir, 'enhanced_word_model.keras')
            if os.path.exists(word_path):
                self.word_model = tf.keras.models.load_model(word_path)
                print("✅ Enhanced word model loaded")
            
            # Load label mappings
            alphabet_labels_path = os.path.join(self.model_dir, 'enhanced_alphabet_labels.json')
            if os.path.exists(alphabet_labels_path):
                with open(alphabet_labels_path, 'r') as f:
                    self.alphabet_labels = json.load(f)
            
            word_labels_path = os.path.join(self.model_dir, 'enhanced_word_labels.json')
            if os.path.exists(word_labels_path):
                with open(word_labels_path, 'r') as f:
                    self.word_labels = json.load(f)
                    
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def preprocess_alphabet_image(self, image):
        """Preprocess image for alphabet prediction"""
        if len(image.shape) == 3:
            image = cv2.resize(image, self.alphabet_img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
        return image
    
    def preprocess_word_video(self, frames):
        """Preprocess video frames for word prediction"""
        processed_frames = []
        
        for frame in frames:
            if frame is not None:
                frame = cv2.resize(frame, self.word_img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                processed_frames.append(frame)
        
        # Pad or trim to exact length
        while len(processed_frames) < self.word_max_frames:
            processed_frames.append(np.zeros((*self.word_img_size, 3)))
        
        return np.array(processed_frames[:self.word_max_frames])
    
    def predict_alphabet(self, image):
        """Predict alphabet from single image"""
        if self.alphabet_model is None:
            return None
            
        processed_image = self.preprocess_alphabet_image(image)
        predictions = self.alphabet_model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_3_indices:
            confidence = float(predictions[0][idx])
            if confidence > 0.01:  # Filter low confidence
                for label, label_idx in self.alphabet_labels.items():
                    if label_idx == idx:
                        results.append({
                            'label': label,
                            'confidence': confidence
                        })
                        break
        
        return results
    
    def predict_word_from_video(self, video_path):
        """Predict word from video file"""
        if self.word_model is None:
            return None
            
        # Extract frames from video
        frames = self.extract_video_frames(video_path)
        if frames is None:
            return None
            
        processed_frames = self.preprocess_word_video(frames)
        processed_frames = np.expand_dims(processed_frames, axis=0)
        
        predictions = self.word_model.predict(processed_frames, verbose=0)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_3_indices:
            confidence = float(predictions[0][idx])
            if confidence > 0.01:
                for word, word_idx in self.word_labels.items():
                    if word_idx == idx:
                        results.append({
                            'word': word,
                            'confidence': confidence
                        })
                        break
        
        return results
    
    def predict_word_from_camera(self, duration=3):
        """Predict word from live camera feed"""
        if self.word_model is None:
            return None
            
        frames = self.capture_video_frames(duration)
        if len(frames) == 0:
            return None
            
        processed_frames = self.preprocess_word_video(frames)
        processed_frames = np.expand_dims(processed_frames, axis=0)
        
        predictions = self.word_model.predict(processed_frames, verbose=0)
        
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx])
        
        for word, word_idx in self.word_labels.items():
            if word_idx == top_idx:
                return {
                    'word': word,
                    'confidence': confidence
                }
        
        return None
    
    def extract_video_frames(self, video_path):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Calculate frame indices to sample
        frames_to_extract = min(total_frames, self.word_max_frames)
        step = max(1, total_frames // frames_to_extract)
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        # Ensure we have the right number of frames
        if len(frames) > self.word_max_frames:
            frames = frames[:self.word_max_frames]
        
        return frames
    
    def capture_video_frames(self, duration=3):
        """Capture frames from camera"""
        cap = cv2.VideoCapture(0)
        frames = []
        fps = 10  # Target FPS
        
        frames_to_capture = duration * fps
        
        for _ in range(frames_to_capture):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            time.sleep(1/fps)
        
        cap.release()
        return frames
    
    def get_alphabet_labels(self):
        """Get alphabet labels mapping"""
        return self.alphabet_labels
    
    def get_word_labels(self):
        """Get word labels mapping"""
        return self.word_labels

# Usage example
if __name__ == "__main__":
    predictor = EnhancedPredictor()
    
    # Test alphabet prediction
    if predictor.alphabet_model:
        print("✅ Alphabet model ready")
        print("Alphabet labels:", predictor.get_alphabet_labels())
    
    # Test word model
    if predictor.word_model:
        print("✅ Word model ready")
        print("Word labels:", predictor.get_word_labels())

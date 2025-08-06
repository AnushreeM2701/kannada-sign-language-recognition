from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from gtts import gTTS
from tempfile import NamedTemporaryFile
import base64
import logging
import time

from scripts.predict_updated import predict_word_sequence
from scripts.cnn_predict_fixed import safe_predict
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
MAX_CAPTURE_FRAMES = 10
CAMERA_TIMEOUT = 30  # seconds
VALID_MODES = ['alphabet', 'word']

# CNN-specific constants
IMG_SIZE = 64
NUM_CHANNELS = 3

# Load CNN models and label maps
MODEL_DIR = "models"
ALPHABET_MODEL_PATH = os.path.join(MODEL_DIR, "alphabet_model.keras")
WORD_MODEL_PATH = os.path.join(MODEL_DIR, "word_model.keras")
ALPHABET_LABEL_MAP_PATH = os.path.join(MODEL_DIR, "alphabet_label_map.json")
WORD_LABEL_MAP_PATH = os.path.join(MODEL_DIR, "word_label_map.json")

# Load models at startup
logger.info("Loading CNN models...")
alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL_PATH)
word_model = tf.keras.models.load_model(WORD_MODEL_PATH)

# Load label maps
with open(ALPHABET_LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    alphabet_label_map = json.load(f)
    alphabet_reverse_map = {v: k for k, v in alphabet_label_map.items()}

with open(WORD_LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    word_label_map = json.load(f)
    word_reverse_map = {v: k for k, v in word_label_map.items()}

logger.info("✅ CNN models loaded successfully")

@app.route('/')
def index():
    return render_template('index.html')

def safe_camera_operation(func):
    """Decorator to ensure camera resources are properly released"""
    def wrapper(*args, **kwargs):
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({'error': 'Cannot open camera. Please check camera permissions.'})
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            return func(cap, *args, **kwargs)
        except Exception as e:
            logger.error(f"Camera operation error: {e}")
            return jsonify({'error': str(e)})
        finally:
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
    return wrapper

@app.route('/test_camera')
def test_camera():
    """Test endpoint to check camera functionality"""
    @safe_camera_operation
    def check_camera(cap):
        ret, frame = cap.read()
        if ret:
            return jsonify({'status': 'success', 'message': 'Camera is working'})
        else:
            return jsonify({'error': 'Cannot read from camera'})
    
    return check_camera()

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for alphabet and word recognition"""
    mode = request.form.get('mode')
    
    # Input validation
    if not mode:
        return jsonify({'error': 'Mode parameter is required'}), 400
    
    if mode not in VALID_MODES:
        return jsonify({'error': f'Invalid mode. Must be one of: {", ".join(VALID_MODES)}'}), 400
    
    @safe_camera_operation
    def perform_prediction(cap):
        try:
            logger.info(f"Starting prediction in {mode} mode")
            
            if mode == 'alphabet':
                logger.info("Capturing single frame for alphabet prediction")
                ret, frame = cap.read()
                
                if not ret:
                    return jsonify({'error': 'Could not read from webcam'}), 500
                
                # Removed cv2.imshow calls to avoid OpenCV GUI errors
                
                result = safe_predict(frame)
                logger.info(f"Alphabet prediction: {result}")
                
            elif mode == 'word':
                logger.info("Capturing frames for word prediction")
                frames = []
                frame_count = 0
                
                for _ in range(MAX_CAPTURE_FRAMES):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    frames.append(frame)
                    frame_count += 1
                    
                    # Removed cv2.imshow calls to avoid OpenCV GUI errors
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                
                if frame_count == 0:
                    return jsonify({'error': 'No frames captured'}), 500
                
                result = predict_word_sequence(frames)
                logger.info(f"Word prediction: {result}")
            
            # Generate TTS audio
            try:
                logger.info("Generating TTS audio")
                tts = gTTS(text=result, lang='kn')
                with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    with open(fp.name, "rb") as audio_file:
                        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
                os.remove(fp.name)
                
                logger.info("Prediction completed successfully")
                return jsonify({
                    'prediction': result, 
                    'audio': audio_base64,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Error generating audio: {e}")
                return jsonify({
                    'prediction': result,
                    'audio': None,
                    'warning': f'Audio generation failed: {str(e)}'
                })
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return perform_prediction()

@app.route('/debug_camera')
def debug_camera():
    """Debug endpoint to check camera availability"""
    try:
        # Check available cameras
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append({
                            'index': i,
                            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'fps': cap.get(cv2.CAP_PROP_FPS)
                        })
                except Exception as e:
                    logger.warning(f"Error testing camera {i}: {e}")
                finally:
                    cap.release()
        
        return jsonify({
            'available_cameras': available_cameras,
            'opencv_version': cv2.__version__,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Debug camera error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Kannada Sign Language Recognition',
        'version': '1.0.0'
    })

if __name__ == "__main__":
    logger.info("Starting Kannada Sign Language Recognition Server")
    app.run(debug=True, host='0.0.0.0', port=5000)

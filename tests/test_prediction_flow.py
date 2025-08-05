import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import pytest
from app_improved import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test the index page loads correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Kannada Sign Language Recognition' in response.data

def test_test_camera_endpoint(client):
    """Test the /test_camera endpoint"""
    response = client.get('/test_camera')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'status' in json_data or 'error' in json_data

def test_health_endpoint(client):
    """Test the /health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data.get('status') == 'healthy'

def test_predict_alphabet_no_mode(client):
    """Test /predict endpoint with no mode parameter"""
    response = client.post('/predict', data={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data

def test_predict_alphabet_invalid_mode(client):
    """Test /predict endpoint with invalid mode"""
    response = client.post('/predict', data={'mode': 'invalid'})
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data

# Note: The following tests require a camera and may fail in headless environments

def test_predict_alphabet_mode(client):
    """Test /predict endpoint with alphabet mode"""
    response = client.post('/predict', data={'mode': 'alphabet'})
    assert response.status_code in (200, 500)  # 500 if no camera available
    json_data = response.get_json()
    assert 'prediction' in json_data or 'error' in json_data

def test_predict_word_mode(client):
    """Test /predict endpoint with word mode"""
    response = client.post('/predict', data={'mode': 'word'})
    assert response.status_code in (200, 500)  # 500 if no camera available
    json_data = response.get_json()
    assert 'prediction' in json_data or 'error' in json_data

#!/usr/bin/env python3
"""
Test script to verify camera functionality
"""
import cv2
import numpy as np

def test_camera():
    """Test basic camera functionality"""
    print("Testing camera access...")
    
    # Try different camera indices
    for camera_index in range(3):
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"✓ Camera {camera_index} opened successfully")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test reading a frame
            ret, frame = cap.read()
            if ret:
                print("✓ Successfully read frame")
                print(f"Frame shape: {frame.shape}")
                
                # Display test window
                cv2.imshow("Camera Test - Press 'q' to quit", frame)
                print("Camera test window opened. Press 'q' to close.")
                
                # Wait for key press
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("✗ Failed to read frame")
                        break
                    
                    cv2.imshow("Camera Test - Press 'q' to quit", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"✗ Could not read frame from camera {camera_index}")
                cap.release()
        else:
            print(f"✗ Could not open camera {camera_index}")
    
    return False

if __name__ == "__main__":
    success = test_camera()
    if success:
        print("✓ Camera test completed successfully!")
    else:
        print("✗ Camera test failed. Please check:")
        print("  1. Camera permissions")
        print("  2. Camera drivers")
        print("  3. Other applications using the camera")

#!/usr/bin/env python3
"""
Quick fix runner for OpenCV C++ exceptions
"""

import os
import subprocess
import sys

def run_diagnostic():
    """Run the diagnostic script"""
    print("🔍 Running diagnostic...")
    try:
        subprocess.run([sys.executable, "diagnose_opencv_error.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Diagnostic failed: {e}")

def run_fixes():
    """Run the fix script"""
    print("🔧 Applying fixes...")
    try:
        subprocess.run([sys.executable, "fix_opencv_exceptions.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Fixes failed: {e}")

def test_camera():
    """Test camera functionality"""
    print("📸 Testing camera...")
    try:
        subprocess.run([sys.executable, "test_camera.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Camera test failed: {e}")

def main():
    print("🚀 OpenCV Exception Fix Runner")
    print("=" * 50)
    
    # Run diagnostic first
    run_diagnostic()
    
    # Apply fixes
    run_fixes()
    
    # Test camera
    test_camera()
    
    print("\n✅ Fix process complete!")
    print("\nNext steps:")
    print("1. Check the diagnostic output above")
    print("2. If camera issues persist, check system permissions")
    print("3. Try running your prediction script again")

if __name__ == "__main__":
    main()

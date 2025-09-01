#!/usr/bin/env python3
"""
AI Image Detector Chrome Extension - Startup Script

This script helps you start the Python backend and provides instructions
for loading the Chrome extension.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'tensorflow', 'scikit-learn', 'opencv-python', 
        'Pillow', 'flask', 'flask-cors', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if trained model files exist"""
    model_files = [
        'model/ai_detector_model.h5',
        'model/scaler.pkl',
        'model/classifier.pkl'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nTraining models with synthetic data...")
        return False
    
    print("✅ All model files found")
    return True

def train_models():
    """Train the models using synthetic data"""
    try:
        print("🔄 Training models...")
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Models trained successfully")
            return True
        else:
            print("❌ Error training models:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running training script: {e}")
        return False

def start_backend():
    """Start the Flask backend server"""
    try:
        print("🚀 Starting Python backend...")
        print("   Backend will be available at: http://localhost:5000")
        print("   Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the Flask app
        subprocess.run([sys.executable, 'ai_detector.py'])
        
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def show_extension_instructions():
    """Show instructions for loading the Chrome extension"""
    print("\n" + "="*60)
    print("📦 CHROME EXTENSION SETUP")
    print("="*60)
    print("1. Open Chrome and go to: chrome://extensions/")
    print("2. Enable 'Developer mode' (toggle in top-right)")
    print("3. Click 'Load unpacked'")
    print("4. Select this project directory")
    print("5. The extension icon should appear in your toolbar")
    print("6. Click the icon to open the popup interface")
    print("\n💡 Tip: Keep this terminal open while using the extension")
    print("="*60)

def main():
    """Main startup function"""
    print("🤖 AI Image Detector Chrome Extension")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model files
    if not check_model_files():
        if not train_models():
            print("❌ Failed to train models. Exiting.")
            return
    
    # Show extension instructions
    show_extension_instructions()
    
    # Ask user if they want to start the backend
    print("\nDo you want to start the Python backend now? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        start_backend()
    else:
        print("\nTo start the backend manually, run:")
        print("python ai_detector.py")
        print("\nMake sure the backend is running before using the extension!")

if __name__ == "__main__":
    main()

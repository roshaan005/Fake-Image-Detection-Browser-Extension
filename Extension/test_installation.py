#!/usr/bin/env python3
"""
Test script for AI Image Detector Chrome Extension

This script tests the installation and functionality of the extension.
"""

import os
import sys
import requests
import json
import base64
import numpy as np
from PIL import Image
import io

def test_backend_health():
    """Test if the backend is running and healthy"""
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend (is it running?)")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a 224x224 test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"

def test_image_prediction():
    """Test image prediction functionality"""
    try:
        # Create test image
        test_image = create_test_image()
        
        # Send prediction request
        response = requests.post('http://localhost:5000/predict', 
                               json={
                                   'image_data': test_image,
                                   'threshold': 0.7
                               },
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                prediction = result.get('result', {})
                print("✅ Image prediction successful")
                print(f"   Is AI: {prediction.get('isAI')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
                print(f"   Deep Learning Score: {prediction.get('deep_learning_score', 0):.3f}")
                print(f"   Random Forest Score: {prediction.get('random_forest_score', 0):.3f}")
                return True
            else:
                print(f"❌ Prediction failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False

def test_extension_files():
    """Test if all extension files exist"""
    required_files = [
        'manifest.json',
        'popup.html',
        'popup.css',
        'popup.js',
        'content.js',
        'background.js'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing extension files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All extension files found")
    return True

def test_manifest_syntax():
    """Test if manifest.json is valid"""
    try:
        with open('manifest.json', 'r') as f:
            manifest = json.load(f)
        
        # Check required fields
        required_fields = ['manifest_version', 'name', 'version', 'permissions']
        missing_fields = []
        
        for field in required_fields:
            if field not in manifest:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required manifest fields: {missing_fields}")
            return False
        
        print("✅ Manifest.json is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in manifest.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading manifest.json: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Testing AI Image Detector Installation")
    print("="*50)
    
    tests = [
        ("Extension Files", test_extension_files),
        ("Manifest Syntax", test_manifest_syntax),
        ("Backend Health", test_backend_health),
        ("Image Prediction", test_image_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is complete.")
        print("\nNext steps:")
        print("1. Load the extension in Chrome (chrome://extensions/)")
        print("2. Navigate to a webpage with images")
        print("3. Click the extension icon to start detecting AI images!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("- Make sure the Python backend is running: python ai_detector.py")
        print("- Check that all dependencies are installed: pip install -r requirements.txt")
        print("- Verify that model files exist in the model/ directory")

if __name__ == "__main__":
    run_all_tests()

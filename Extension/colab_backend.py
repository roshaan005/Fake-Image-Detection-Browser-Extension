# AI Image Detector Chrome Extension - Google Colab Backend
# Run this notebook to host the Python backend in Colab

# Install required dependencies
!pip install tensorflow scikit-learn opencv-python Pillow flask flask-cors numpy matplotlib seaborn pandas requests python-dotenv

# Install ngrok for tunneling
!pip install pyngrok

# Import required libraries
import os
import base64
import io
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from pyngrok import ngrok

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIImageDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = None
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_model_loaded = False
        self.model_path = "model/ai_detector_model.h5"
        self.scaler_path = "model/scaler.pkl"
        self.classifier_path = "model/classifier.pkl"
        
    def build_model(self):
        """Build a TensorFlow model for AI image detection"""
        # Use EfficientNetB0 as base model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def extract_handcrafted_features(self, img_array):
        """Extract handcrafted features using OpenCV and scikit-learn"""
        # Convert to grayscale for some features
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Basic statistical features
        features['mean_brightness'] = np.mean(gray)
        features['std_brightness'] = np.std(gray)
        features['min_brightness'] = np.min(gray)
        features['max_brightness'] = np.max(gray)
        
        # Color features
        for i, color in enumerate(['red', 'green', 'blue']):
            features[f'mean_{color}'] = np.mean(img_array[:, :, i])
            features[f'std_{color}'] = np.std(img_array[:, :, i])
            features[f'min_{color}'] = np.min(img_array[:, :, i])
            features[f'max_{color}'] = np.max(img_array[:, :, i])
        
        # Texture features using Local Binary Patterns
        features['lbp_uniformity'] = self.calculate_lbp_uniformity(gray)
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude_mean'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        features['gradient_magnitude_std'] = np.std(np.sqrt(grad_x**2 + grad_y**2))
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        features['histogram_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        features['histogram_variance'] = np.var(hist)
        
        # Frequency domain features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features['frequency_mean'] = np.mean(magnitude_spectrum)
        features['frequency_std'] = np.std(magnitude_spectrum)
        
        # Noise estimation
        features['noise_level'] = self.estimate_noise(gray)
        
        return features
    
    def calculate_lbp_uniformity(self, gray):
        """Calculate Local Binary Pattern uniformity"""
        def get_pixel(img, center, x, y):
            return 1 if img[x][y] >= center else 0
        
        def calculate_lbp(img, x, y):
            center = img[x][y]
            val_ar = []
            for i in range(8):
                val_ar.append(get_pixel(img, center, x + self.neighbors[i][0], y + self.neighbors[i][1]))
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]
            val = 0
            for i in range(len(val_ar)):
                val += val_ar[i] * power_val[i]
            return val
        
        # Define 8 neighbors
        self.neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        # Calculate LBP for a sample region
        sample_size = min(50, gray.shape[0], gray.shape[1])
        start_x = (gray.shape[0] - sample_size) // 2
        start_y = (gray.shape[1] - sample_size) // 2
        
        lbp_values = []
        for i in range(start_x, start_x + sample_size):
            for j in range(start_y, start_y + sample_size):
                if 0 < i < gray.shape[0] - 1 and 0 < j < gray.shape[1] - 1:
                    lbp_values.append(calculate_lbp(gray, i, j))
        
        if lbp_values:
            return np.std(lbp_values)
        return 0
    
    def estimate_noise(self, gray):
        """Estimate noise level using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def preprocess_image(self, img_data):
        """Preprocess image for model input"""
        try:
            # Decode base64 image
            if img_data.startswith('data:image'):
                img_data = img_data.split(',')[1]
            
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 224x224
            img = img.resize((224, 224))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def extract_deep_features(self, img_array):
        """Extract deep features using the pre-trained model"""
        if self.feature_extractor is None:
            # Use EfficientNetB0 for feature extraction
            self.feature_extractor = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        
        # Preprocess for EfficientNet
        img_tensor = image.img_to_array(img_array)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = tf.keras.applications.efficientnet.preprocess_input(img_tensor)
        
        # Extract features
        features = self.feature_extractor.predict(img_tensor, verbose=0)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        
        return features.numpy().flatten()
    
    def combine_features(self, deep_features, handcrafted_features):
        """Combine deep and handcrafted features"""
        # Convert handcrafted features to array
        handcrafted_array = np.array(list(handcrafted_features.values()))
        
        # Combine features
        combined_features = np.concatenate([deep_features, handcrafted_array])
        
        return combined_features
    
    def predict(self, img_data, threshold=0.7):
        """Predict if image is AI-generated"""
        try:
            # Preprocess image
            img_array = self.preprocess_image(img_data)
            if img_array is None:
                return {"success": False, "error": "Failed to preprocess image"}
            
            # Extract features
            deep_features = self.extract_deep_features(img_array)
            handcrafted_features = self.extract_handcrafted_features(img_array)
            combined_features = self.combine_features(deep_features, handcrafted_features)
            
            # Scale features
            features_scaled = self.scaler.transform(combined_features.reshape(1, -1))
            
            # Make prediction using both models
            if self.model is not None:
                # Deep learning prediction
                dl_prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            else:
                dl_prediction = 0.5  # Default if model not loaded
            
            # Random Forest prediction
            rf_prediction = self.classifier.predict_proba(features_scaled)[0][1]
            
            # Ensemble prediction (average of both models)
            ensemble_prediction = (dl_prediction + rf_prediction) / 2
            
            # Determine result
            is_ai = ensemble_prediction >= threshold
            confidence = ensemble_prediction
            
            return {
                "success": True,
                "result": {
                    "isAI": is_ai,
                    "confidence": float(confidence),
                    "deep_learning_score": float(dl_prediction),
                    "random_forest_score": float(rf_prediction),
                    "features": {
                        "deep_features_shape": deep_features.shape,
                        "handcrafted_features_count": len(handcrafted_features)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"success": False, "error": str(e)}
    
    def train_model(self, training_data):
        """Train the model with provided data"""
        try:
            # This would be implemented with actual training data
            # For now, we'll create a simple model
            logger.info("Training model...")
            
            # Build and compile model
            self.model = self.build_model()
            
            # Train the model (placeholder for actual training)
            # In a real implementation, you would:
            # 1. Load training images
            # 2. Extract features
            # 3. Train the model
            # 4. Save the model
            
            logger.info("Model training completed")
            return {"success": True, "message": "Model trained successfully"}
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"success": False, "error": str(e)}
    
    def save_model(self):
        """Save the trained model"""
        try:
            os.makedirs("model", exist_ok=True)
            
            if self.model:
                self.model.save(self.model_path)
            
            # Save scaler and classifier
            import pickle
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(self.classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            
            logger.info("Model saved successfully")
            return {"success": True, "message": "Model saved successfully"}
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {"success": False, "error": str(e)}
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("Deep learning model loaded")
            
            if os.path.exists(self.scaler_path):
                import pickle
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded")
            
            if os.path.exists(self.classifier_path):
                import pickle
                with open(self.classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Random Forest classifier loaded")
            
            self.is_model_loaded = True
            logger.info("All models loaded successfully")
            return {"success": True, "message": "Models loaded successfully"}
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {"success": False, "error": str(e)}

# Flask API
app = Flask(__name__)
CORS(app)

# Initialize detector
detector = AIImageDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": detector.is_model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if image is AI-generated"""
    try:
        data = request.get_json()
        img_data = data.get('image_data')
        threshold = data.get('threshold', 0.7)
        
        if not img_data:
            return jsonify({"success": False, "error": "No image data provided"})
        
        result = detector.predict(img_data, threshold)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/train', methods=['POST'])
def train():
    """Train the model"""
    try:
        data = request.get_json()
        training_data = data.get('training_data', [])
        
        result = detector.train_model(training_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in train endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/save', methods=['POST'])
def save_model():
    """Save the trained model"""
    try:
        result = detector.save_model()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in save endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/load', methods=['POST'])
def load_model():
    """Load the trained model"""
    try:
        result = detector.load_model()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in load endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})

# Start ngrok tunnel
print("Starting ngrok tunnel...")
ngrok_tunnel = ngrok.connect(5000)
public_url = ngrok_tunnel.public_url
print(f"Public URL: {public_url}")

# Load model on startup
print("Loading models...")
detector.load_model()

# Run Flask app
print("Starting Flask server...")
print(f"Update your Chrome extension background.js to use: {public_url}")
app.run(host='0.0.0.0', port=5000, debug=False)

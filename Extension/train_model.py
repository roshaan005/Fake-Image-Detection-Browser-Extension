import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from ai_detector import AIImageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.detector = AIImageDetector()
        self.training_data = []
        self.validation_data = []
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data for demonstration"""
        logger.info(f"Generating {num_samples} synthetic training samples...")
        
        # Create directories
        os.makedirs("training_data/real", exist_ok=True)
        os.makedirs("training_data/ai", exist_ok=True)
        
        # Generate synthetic AI-like images (simplified)
        for i in range(num_samples // 2):
            # Generate synthetic AI image (simplified - in reality, you'd use actual AI-generated images)
            img = self.generate_synthetic_ai_image()
            img_path = f"training_data/ai/ai_{i:04d}.png"
            img.save(img_path)
            self.training_data.append({
                'path': img_path,
                'label': 1,  # AI-generated
                'type': 'ai'
            })
        
        # Generate synthetic real-like images
        for i in range(num_samples // 2):
            img = self.generate_synthetic_real_image()
            img_path = f"training_data/real/real_{i:04d}.png"
            img.save(img_path)
            self.training_data.append({
                'path': img_path,
                'label': 0,  # Real
                'type': 'real'
            })
        
        logger.info("Synthetic data generation completed")
    
    def generate_synthetic_ai_image(self):
        """Generate a synthetic AI-like image"""
        # Create a 224x224 image with AI-like characteristics
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some AI-like patterns (simplified)
        # Add gradients
        for i in range(224):
            for j in range(224):
                img_array[i, j, 0] = int(255 * (i + j) / (224 + 224))  # Red gradient
                img_array[i, j, 1] = int(255 * (i - j + 224) / (224 + 224))  # Green gradient
                img_array[i, j, 2] = int(255 * (j - i + 224) / (224 + 224))  # Blue gradient
        
        # Add some noise
        noise = np.random.normal(0, 20, (224, 224, 3))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Add some geometric patterns
        cv2.circle(img_array, (112, 112), 50, (255, 255, 255), 2)
        
        return Image.fromarray(img_array)
    
    def generate_synthetic_real_image(self):
        """Generate a synthetic real-like image"""
        # Create a more natural-looking image
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add natural variations
        for i in range(224):
            for j in range(224):
                # Add some natural color variations
                img_array[i, j, 0] = int(img_array[i, j, 0] + np.random.normal(0, 30))
                img_array[i, j, 1] = int(img_array[i, j, 1] + np.random.normal(0, 30))
                img_array[i, j, 2] = int(img_array[i, j, 2] + np.random.normal(0, 30))
        
        # Add more realistic noise
        noise = np.random.normal(0, 15, (224, 224, 3))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Add some blur to simulate real camera characteristics
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
        
        return Image.fromarray(img_array)
    
    def extract_features_from_dataset(self):
        """Extract features from the training dataset"""
        logger.info("Extracting features from training dataset...")
        
        features_list = []
        labels_list = []
        
        for item in self.training_data:
            try:
                # Load image
                img = Image.open(item['path'])
                img_array = np.array(img)
                
                # Extract features
                deep_features = self.detector.extract_deep_features(img_array)
                handcrafted_features = self.detector.extract_handcrafted_features(img_array)
                combined_features = self.detector.combine_features(deep_features, handcrafted_features)
                
                features_list.append(combined_features)
                labels_list.append(item['label'])
                
            except Exception as e:
                logger.warning(f"Error processing {item['path']}: {e}")
                continue
        
        logger.info(f"Extracted features from {len(features_list)} images")
        return np.array(features_list), np.array(labels_list)
    
    def train_models(self):
        """Train both TensorFlow and scikit-learn models"""
        logger.info("Starting model training...")
        
        # Extract features
        features, labels = self.extract_features_from_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        self.detector.scaler.fit(X_train)
        X_train_scaled = self.detector.scaler.transform(X_train)
        X_test_scaled = self.detector.scaler.transform(X_test)
        
        # Train Random Forest classifier
        logger.info("Training Random Forest classifier...")
        self.detector.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate Random Forest
        rf_pred = self.detector.classifier.predict(X_test_scaled)
        rf_proba = self.detector.classifier.predict_proba(X_test_scaled)
        
        logger.info("Random Forest Results:")
        logger.info(classification_report(y_test, rf_pred))
        
        # Train TensorFlow model
        logger.info("Training TensorFlow model...")
        self.detector.model = self.detector.build_model()
        
        # Prepare data for TensorFlow
        # We'll use the deep features only for the TensorFlow model
        deep_features_train = X_train[:, :1280]  # EfficientNetB0 features
        deep_features_test = X_test[:, :1280]
        
        # Normalize deep features
        deep_features_train = deep_features_train / 255.0
        deep_features_test = deep_features_test / 255.0
        
        # Train the model
        history = self.detector.model.fit(
            deep_features_train, y_train,
            validation_data=(deep_features_test, y_test),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate TensorFlow model
        tf_pred = (self.detector.model.predict(deep_features_test) > 0.5).astype(int)
        
        logger.info("TensorFlow Model Results:")
        logger.info(classification_report(y_test, tf_pred))
        
        # Save models
        self.save_models()
        
        # Plot training history
        self.plot_training_history(history)
        
        logger.info("Model training completed successfully")
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving trained models...")
        
        os.makedirs("model", exist_ok=True)
        
        # Save TensorFlow model
        if self.detector.model:
            self.detector.model.save("model/ai_detector_model.h5")
            logger.info("TensorFlow model saved")
        
        # Save scikit-learn models
        with open("model/scaler.pkl", 'wb') as f:
            pickle.dump(self.detector.scaler, f)
        logger.info("Scaler saved")
        
        with open("model/classifier.pkl", 'wb') as f:
            pickle.dump(self.detector.classifier, f)
        logger.info("Random Forest classifier saved")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Plot loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Plot recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training history plot saved")
    
    def evaluate_models(self):
        """Evaluate the trained models"""
        logger.info("Evaluating models...")
        
        # Load test data
        features, labels = self.extract_features_from_dataset()
        _, X_test, _, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_test_scaled = self.detector.scaler.transform(X_test)
        
        # Evaluate Random Forest
        rf_pred = self.detector.classifier.predict(X_test_scaled)
        rf_proba = self.detector.classifier.predict_proba(X_test_scaled)
        
        # Evaluate TensorFlow model
        deep_features_test = X_test[:, :1280] / 255.0
        tf_pred = (self.detector.model.predict(deep_features_test) > 0.5).astype(int)
        tf_proba = self.detector.model.predict(deep_features_test)
        
        # Print results
        logger.info("Random Forest Results:")
        logger.info(classification_report(y_test, rf_pred))
        
        logger.info("TensorFlow Model Results:")
        logger.info(classification_report(y_test, tf_pred))
        
        # Plot confusion matrices
        self.plot_confusion_matrices(y_test, rf_pred, tf_pred)
    
    def plot_confusion_matrices(self, y_true, rf_pred, tf_pred):
        """Plot confusion matrices for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Random Forest confusion matrix
        cm_rf = confusion_matrix(y_true, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Random Forest Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # TensorFlow confusion matrix
        cm_tf = confusion_matrix(y_true, tf_pred)
        sns.heatmap(cm_tf, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('TensorFlow Model Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('model/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Confusion matrices saved")

def main():
    """Main training function"""
    trainer = ModelTrainer()
    
    # Generate synthetic data
    trainer.generate_synthetic_data(num_samples=1000)
    
    # Train models
    trainer.train_models()
    
    # Evaluate models
    trainer.evaluate_models()
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()

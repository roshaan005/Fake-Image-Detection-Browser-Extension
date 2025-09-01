class AIImageDetectorBackground {
    constructor() {
        this.pythonBackendUrl = 'http://localhost:5001';
        this.isModelLoaded = false;
        this.init();
    }

    async init() {
        this.setupMessageListener();
        await this.checkBackendHealth();
    }

    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'analyzeImage') {
                this.analyzeImage(request.imageData, request.threshold).then(sendResponse);
                return true;
            } else if (request.action === 'analyzeFeatures') {
                this.analyzeFeatures(request.features, request.threshold).then(sendResponse);
                return true;
            }
        });
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.pythonBackendUrl}/health`, {
                method: 'GET',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.isModelLoaded = data.model_loaded;
                console.log('Python backend is healthy, model loaded:', this.isModelLoaded);
            } else {
                console.warn('Python backend is not healthy');
                this.isModelLoaded = false;
            }
        } catch (error) {
            console.error('Error checking backend health:', error);
            this.isModelLoaded = false;
        }
    }



    async analyzeImage(imageData, threshold = 70) {
        if (!this.isModelLoaded) {
            return { success: false, error: 'Python backend not available' };
        }

        try {
            const response = await fetch(`${this.pythonBackendUrl}/predict`, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_data: imageData,
                    threshold: threshold / 100
                })
            });

            const result = await response.json();
            
            if (result.success) {
                // Add analysis method indicator
                result.result.analysisMethod = 'full';
            }
            
            return result;
        } catch (error) {
            console.error('Error analyzing image:', error);
            return { success: false, error: error.message };
        }
    }

    async analyzeFeatures(features, threshold = 70) {
        if (!this.isModelLoaded) {
            return { success: false, error: 'Python backend not available' };
        }

        try {
            // Create a simple feature representation for the Python backend
            // Handle both canvas-extracted features and fallback features
            const pixelData = features.pixelData || {};
            const urlFeatures = features.urlFeatures || {};
            const dimensionFeatures = features.dimensionFeatures || {};
            
            const featureData = {
                brightness: pixelData.brightness || urlFeatures.brightness || 128,
                contrast: pixelData.contrast || urlFeatures.contrast || 50,
                edgeDensity: pixelData.edgeDensity || urlFeatures.edgeDensity || 0.1,
                avgRed: pixelData.avgRed || urlFeatures.avgRed || 128,
                avgGreen: pixelData.avgGreen || urlFeatures.avgGreen || 128,
                avgBlue: pixelData.avgBlue || urlFeatures.avgBlue || 128,
                width: features.width || 224,
                height: features.height || 224,
                // Add dimension-based features
                aspectRatio: features.aspectRatio || 1,
                isHighRes: dimensionFeatures.isHighRes || false,
                isStandardRatio: dimensionFeatures.isStandardRatio || false
            };
            
            const response = await fetch(`${this.pythonBackendUrl}/predict_features`, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    features: featureData,
                    threshold: threshold / 100
                })
            });

            const result = await response.json();
            if (result.success) {
                result.result.imageUrl = features.metadata?.src || 'unknown';
                // Add fallback indicator
                if (urlFeatures.brightness) {
                    result.result.analysisMethod = 'fallback';
                } else {
                    result.result.analysisMethod = 'canvas';
                }
            }
            return result;
        } catch (error) {
            console.error('Error analyzing features:', error);
            return { success: false, error: error.message };
        }
    }


}

// Initialize background service worker
new AIImageDetectorBackground();

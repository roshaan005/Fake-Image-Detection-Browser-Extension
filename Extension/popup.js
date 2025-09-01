class AIImageDetectorPopup {
    constructor() {
        this.elements = {};
        this.settings = {};
        this.init();
    }

    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadSettings();
        this.updateStatus('Ready');
    }

    cacheElements() {
        this.elements = {
            scanPage: document.getElementById('scanPage'),
            scanImage: document.getElementById('scanImage'),
            uploadArea: document.getElementById('uploadArea'),
            uploadBox: document.getElementById('uploadBox'),
            imageInput: document.getElementById('imageInput'),
            results: document.getElementById('results'),
            confidenceBar: document.getElementById('confidenceBar'),
            confidenceText: document.getElementById('confidenceText'),
            verdict: document.getElementById('verdict'),
            verdictIcon: document.getElementById('verdictIcon'),
            verdictText: document.getElementById('verdictText'),
            autoScan: document.getElementById('autoScan'),
            confidenceThreshold: document.getElementById('confidenceThreshold'),
            thresholdValue: document.getElementById('thresholdValue'),
            status: document.getElementById('status'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText')
        };
    }

    bindEvents() {
        this.elements.scanPage.addEventListener('click', () => this.scanCurrentPage());
        this.elements.scanImage.addEventListener('click', () => this.toggleUploadArea());
        this.elements.uploadBox.addEventListener('click', () => this.elements.imageInput.click());
        this.elements.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.elements.autoScan.addEventListener('change', (e) => this.saveSetting('autoScan', e.target.checked));
        this.elements.confidenceThreshold.addEventListener('input', (e) => this.updateThreshold(e.target.value));

        // Drag and drop support
        this.elements.uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadBox.style.borderColor = '#667eea';
            this.elements.uploadBox.style.background = '#f8f9ff';
        });

        this.elements.uploadBox.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.elements.uploadBox.style.borderColor = '#dee2e6';
            this.elements.uploadBox.style.background = 'white';
        });

        this.elements.uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadBox.style.borderColor = '#dee2e6';
            this.elements.uploadBox.style.background = 'white';
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.processImage(files[0]);
            }
        });
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get(['autoScan', 'confidenceThreshold']);
            this.settings = {
                autoScan: result.autoScan ?? false,
                confidenceThreshold: result.confidenceThreshold ?? 70
            };
            
            this.elements.autoScan.checked = this.settings.autoScan;
            this.elements.confidenceThreshold.value = this.settings.confidenceThreshold;
            this.elements.thresholdValue.textContent = `${this.settings.confidenceThreshold}%`;
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    async saveSetting(key, value) {
        try {
            await chrome.storage.sync.set({ [key]: value });
            this.settings[key] = value;
        } catch (error) {
            console.error('Error saving setting:', error);
        }
    }

    updateThreshold(value) {
        this.elements.thresholdValue.textContent = `${value}%`;
        this.saveSetting('confidenceThreshold', parseInt(value));
    }

    toggleUploadArea() {
        const isVisible = this.elements.uploadArea.style.display !== 'none';
        this.elements.uploadArea.style.display = isVisible ? 'none' : 'block';
        
        if (!isVisible) {
            this.elements.scanImage.textContent = 'Cancel Upload';
            this.elements.scanImage.classList.add('secondary');
        } else {
            this.elements.scanImage.innerHTML = '<span class="icon">📷</span>Scan Uploaded Image';
            this.elements.scanImage.classList.remove('secondary');
        }
    }

    async scanCurrentPage() {
        this.updateStatus('Scanning page...', 'loading');
        this.elements.scanPage.disabled = true;

        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            // Check if we can inject the content script
            if (!tab.url.startsWith('http')) {
                throw new Error('Cannot scan this type of page (chrome://, file://, etc.)');
            }
            
            // Try to send message to content script
            let response;
            try {
                response = await chrome.tabs.sendMessage(tab.id, {
                    action: 'scanImages',
                    threshold: this.settings.confidenceThreshold
                });
            } catch (error) {
                // Content script not available, inject it first
                console.log('Content script not found, injecting...');
                await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ['content.js']
                });
                
                // Wait a moment for the script to initialize
                await new Promise(resolve => setTimeout(resolve, 500));
                
                // Try again
                response = await chrome.tabs.sendMessage(tab.id, {
                    action: 'scanImages',
                    threshold: this.settings.confidenceThreshold
                });
            }

            if (response && response.success) {
                this.showResults(response.results);
                const totalImages = response.totalImages || response.results.length;
                const analyzedImages = response.analyzedImages || response.results.length;
                const skippedImages = response.skippedImages || 0;
                
                // Count fallback vs canvas analysis
                const fallbackCount = response.results.filter(r => r.analysisMethod === 'fallback').length;
                const canvasCount = response.results.filter(r => r.analysisMethod === 'canvas').length;
                
                let statusMessage = `Found ${analyzedImages} analyzable images`;
                if (fallbackCount > 0) {
                    statusMessage += ` (${fallbackCount} using fallback analysis)`;
                }
                if (skippedImages > 0) {
                    statusMessage += `, ${skippedImages} skipped`;
                }
                
                this.updateStatus(statusMessage);
            } else {
                throw new Error('Failed to scan page');
            }
        } catch (error) {
            console.error('Error scanning page:', error);
            this.updateStatus('Error scanning page', 'error');
        } finally {
            this.elements.scanPage.disabled = false;
        }
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processImage(file);
        }
    }

    async processImage(file) {
        this.updateStatus('Processing image...', 'loading');
        this.elements.scanImage.disabled = true;

        try {
            const base64 = await this.fileToBase64(file);
            
            // Send to background script for processing
            const response = await chrome.runtime.sendMessage({
                action: 'analyzeImage',
                imageData: base64,
                threshold: this.settings.confidenceThreshold
            });

            if (response && response.success) {
                this.showResults([response.result]);
                const analysisMethod = response.result.analysisMethod || 'unknown';
                this.updateStatus(`Analysis complete (${analysisMethod} analysis)`);
            } else {
                const errorMsg = response?.error || 'Failed to analyze image';
                throw new Error(errorMsg);
            }
        } catch (error) {
            console.error('Error processing image:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
        } finally {
            this.elements.scanImage.disabled = false;
            this.toggleUploadArea(); // Hide upload area
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    showResults(results) {
        if (results.length === 0) {
            this.elements.results.style.display = 'none';
            return;
        }

        const result = results[0]; // Show first result for now
        const confidence = Math.round(result.confidence * 100);
        
        // Update confidence bar
        this.elements.confidenceBar.style.width = `${confidence}%`;
        this.elements.confidenceText.textContent = `${confidence}%`;

        // Update verdict
        let verdictClass = 'uncertain';
        let verdictIcon = '❓';
        let verdictText = 'Uncertain';

        if (confidence >= this.settings.confidenceThreshold) {
            if (result.isAI) {
                verdictClass = 'ai';
                verdictIcon = '🤖';
                verdictText = 'AI Generated';
            } else {
                verdictClass = 'real';
                verdictIcon = '✅';
                verdictText = 'Real Image';
            }
        }

        this.elements.verdict.className = `verdict ${verdictClass}`;
        this.elements.verdictIcon.textContent = verdictIcon;
        this.elements.verdictText.textContent = verdictText;

        this.elements.results.style.display = 'block';
    }

    updateStatus(text, type = 'ready') {
        this.elements.statusText.textContent = text;
        this.elements.statusIndicator.className = `status-indicator ${type}`;
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AIImageDetectorPopup();
});

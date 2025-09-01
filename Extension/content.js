class AIImageDetectorContent {
    constructor() {
        this.analyzedImages = new Set();
        this.observer = null;
        this.init();
    }

    init() {
        console.log('AI Image Detector Content Script initialized');
        this.setupMessageListener();
        this.setupImageObserver();
        this.scanExistingImages();
    }

    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            console.log('Content script received message:', request);
            if (request.action === 'scanImages') {
                this.scanAllImages(request.threshold).then(sendResponse);
                return true; // Keep message channel open for async response
            }
        });
    }

    setupImageObserver() {
        // Observe for new images being added to the page
        this.observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        let images = [];
                        if (node.querySelectorAll) {
                            images = Array.from(node.querySelectorAll('img'));
                        }
                        if (node.tagName === 'IMG') {
                            images.push(node);
                        }
                        
                        images.forEach(img => {
                            if (!this.analyzedImages.has(img)) {
                                this.analyzeImage(img);
                            }
                        });
                    }
                });
            });
        });

        this.observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    async scanExistingImages() {
        const images = document.querySelectorAll('img');
        const results = [];
        
        for (const img of images) {
            if (!this.analyzedImages.has(img)) {
                const result = await this.analyzeImage(img);
                if (result) {
                    results.push(result);
                }
            }
        }

        return results;
    }

    async scanAllImages(threshold = 70) {
        try {
            console.log('Starting image scan with threshold:', threshold);
            const images = document.querySelectorAll('img');
            console.log('Found', images.length, 'images on page');
            
            const results = [];
            let skippedCount = 0;
            
            for (const img of images) {
                if (img.src && img.complete && img.naturalWidth > 0) {
                    console.log('Analyzing image:', img.src);
                    const result = await this.analyzeImage(img, threshold);
                    if (result) {
                        results.push(result);
                    } else {
                        skippedCount++;
                    }
                }
            }

            console.log('Scan complete, found', results.length, 'analyzable images, skipped', skippedCount);
            return {
                success: true,
                results: results,
                totalImages: images.length,
                analyzedImages: results.length,
                skippedImages: skippedCount
            };
        } catch (error) {
            console.error('Error in scanAllImages:', error);
            return {
                success: false,
                error: error.message,
                results: []
            };
        }
    }

    async analyzeImage(img, threshold = 70) {
        if (this.analyzedImages.has(img)) {
            return null;
        }

        this.analyzedImages.add(img);

        try {
            // Skip small images and data URLs
            if (img.naturalWidth < 100 || img.naturalHeight < 100 || img.src.startsWith('data:')) {
                console.log('Skipping small image or data URL:', img.src);
                return null;
            }

            // Extract image features (with CORS fallback)
            const features = await this.extractImageFeatures(img);
            
            if (!features) {
                console.log('Could not extract features from image:', img.src);
                return null;
            }
            
            // Send to background script for analysis
            const response = await chrome.runtime.sendMessage({
                action: 'analyzeFeatures',
                features: features,
                imageUrl: img.src,
                threshold: threshold
            });

            if (response && response.success) {
                this.markImage(img, response.result);
                return response.result;
            }
        } catch (error) {
            console.error('Error analyzing image:', error);
            // Remove from analyzed set so we can retry later
            this.analyzedImages.delete(img);
        }

        return null;
    }

    async extractImageFeatures(img) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size to a reasonable size for feature extraction
            const maxSize = 224;
            let { width, height } = img;
            
            if (width > height) {
                height = (height * maxSize) / width;
                width = maxSize;
            } else {
                width = (width * maxSize) / height;
                height = maxSize;
            }
            
            canvas.width = width;
            canvas.height = height;
            
            try {
                ctx.drawImage(img, 0, 0, width, height);
                
                // Get image data
                const imageData = ctx.getImageData(0, 0, width, height);
                const data = imageData.data;
                
                // Extract basic features
                const features = {
                    width: width,
                    height: height,
                    aspectRatio: width / height,
                    pixelData: this.extractPixelFeatures(data, width, height),
                    metadata: this.extractMetadata(img)
                };
                
                resolve(features);
            } catch (error) {
                console.log('CORS error drawing image to canvas, using fallback features:', img.src);
                // Fallback: extract features from image metadata and URL patterns
                const fallbackFeatures = this.extractFallbackFeatures(img);
                resolve(fallbackFeatures);
            }
        });
    }

    extractPixelFeatures(data, width, height) {
        const features = {
            brightness: 0,
            contrast: 0,
            saturation: 0,
            edgeDensity: 0,
            colorHistogram: new Array(256).fill(0),
            redHistogram: new Array(256).fill(0),
            greenHistogram: new Array(256).fill(0),
            blueHistogram: new Array(256).fill(0)
        };

        let totalBrightness = 0;
        let totalRed = 0, totalGreen = 0, totalBlue = 0;
        let pixelCount = 0;

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Calculate brightness
            const brightness = (r + g + b) / 3;
            totalBrightness += brightness;
            
            // Update histograms
            features.colorHistogram[Math.floor(brightness)]++;
            features.redHistogram[r]++;
            features.greenHistogram[g]++;
            features.blueHistogram[b]++;
            
            totalRed += r;
            totalGreen += g;
            totalBlue += b;
            pixelCount++;
        }

        // Calculate average brightness
        features.brightness = totalBrightness / pixelCount;
        
        // Calculate average RGB values
        features.avgRed = totalRed / pixelCount;
        features.avgGreen = totalGreen / pixelCount;
        features.avgBlue = totalBlue / pixelCount;

        // Calculate contrast (standard deviation of brightness)
        let variance = 0;
        for (let i = 0; i < data.length; i += 4) {
            const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
            variance += Math.pow(brightness - features.brightness, 2);
        }
        features.contrast = Math.sqrt(variance / pixelCount);

        // Calculate edge density (simplified)
        features.edgeDensity = this.calculateEdgeDensity(data, width, height);

        return features;
    }

    extractMetadata(img) {
        return {
            src: img.src,
            alt: img.alt || '',
            className: img.className || '',
            naturalWidth: img.naturalWidth,
            naturalHeight: img.naturalHeight,
            loading: img.loading || 'auto'
        };
    }

    calculateEdgeDensity(data, width, height) {
        let edgePixels = 0;
        const threshold = 30;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                
                // Simple edge detection using horizontal and vertical gradients
                const hGradient = Math.abs(data[idx] - data[idx + 4]) + 
                                 Math.abs(data[idx + 1] - data[idx + 5]) + 
                                 Math.abs(data[idx + 2] - data[idx + 6]);
                
                const vGradient = Math.abs(data[idx] - data[idx + width * 4]) + 
                                 Math.abs(data[idx + 1] - data[idx + width * 4 + 1]) + 
                                 Math.abs(data[idx + 2] - data[idx + width * 4 + 2]);
                
                if (hGradient > threshold || vGradient > threshold) {
                    edgePixels++;
                }
            }
        }

        return edgePixels / (width * height);
    }

    extractFallbackFeatures(img) {
        // Extract features from image metadata and URL patterns when CORS blocks canvas access
        const url = img.src;
        const width = img.naturalWidth || img.width;
        const height = img.naturalHeight || img.height;
        const aspectRatio = width / height;
        
        // Analyze URL patterns for common AI image indicators
        const urlFeatures = this.analyzeUrlPatterns(url);
        
        // Analyze image dimensions and aspect ratio
        const dimensionFeatures = this.analyzeDimensions(width, height, aspectRatio);
        
        // Create fallback pixel data based on URL and dimension analysis
        const pixelData = {
            brightness: urlFeatures.brightness,
            contrast: urlFeatures.contrast,
            saturation: urlFeatures.saturation,
            edgeDensity: urlFeatures.edgeDensity,
            colorHistogram: new Array(256).fill(0),
            redHistogram: new Array(256).fill(0),
            greenHistogram: new Array(256).fill(0),
            blueHistogram: new Array(256).fill(0),
            avgRed: urlFeatures.avgRed,
            avgGreen: urlFeatures.avgGreen,
            avgBlue: urlFeatures.avgBlue
        };
        
        return {
            width: width,
            height: height,
            aspectRatio: aspectRatio,
            pixelData: pixelData,
            metadata: this.extractMetadata(img),
            urlFeatures: urlFeatures,
            dimensionFeatures: dimensionFeatures
        };
    }

    analyzeUrlPatterns(url) {
        const urlLower = url.toLowerCase();
        
        // Default values
        let brightness = 128;
        let contrast = 50;
        let saturation = 0.5;
        let edgeDensity = 0.1;
        let avgRed = 128, avgGreen = 128, avgBlue = 128;
        
        // Analyze URL for common patterns
        if (urlLower.includes('unsplash')) {
            // Unsplash images are typically high-quality, well-lit photos
            brightness = 140;
            contrast = 60;
            saturation = 0.6;
            edgeDensity = 0.15;
            avgRed = 135; avgGreen = 140; avgBlue = 145;
        } else if (urlLower.includes('pexels')) {
            // Pexels images are also high-quality
            brightness = 135;
            contrast = 55;
            saturation = 0.55;
            edgeDensity = 0.12;
            avgRed = 130; avgGreen = 135; avgBlue = 140;
        } else if (urlLower.includes('pixabay')) {
            brightness = 130;
            contrast = 52;
            saturation = 0.52;
            edgeDensity = 0.11;
            avgRed = 128; avgGreen = 132; avgBlue = 138;
        } else if (urlLower.includes('ai') || urlLower.includes('generated') || urlLower.includes('synthetic')) {
            // Potential AI-generated images
            brightness = 120;
            contrast = 45;
            saturation = 0.4;
            edgeDensity = 0.08;
            avgRed = 125; avgGreen = 125; avgBlue = 125;
        }
        
        return {
            brightness: brightness,
            contrast: contrast,
            saturation: saturation,
            edgeDensity: edgeDensity,
            avgRed: avgRed,
            avgGreen: avgGreen,
            avgBlue: avgBlue
        };
    }

    analyzeDimensions(width, height, aspectRatio) {
        // Analyze dimensions for common AI image patterns
        const features = {
            isSquare: Math.abs(aspectRatio - 1) < 0.1,
            isPortrait: aspectRatio < 0.8,
            isLandscape: aspectRatio > 1.2,
            isStandardRatio: Math.abs(aspectRatio - 16/9) < 0.1 || Math.abs(aspectRatio - 4/3) < 0.1,
            resolution: width * height,
            isHighRes: width * height > 1000000, // > 1MP
            isVeryHighRes: width * height > 4000000 // > 4MP
        };
        
        return features;
    }

    markImage(img, result) {
        const confidence = Math.round(result.confidence * 100);
        const isAI = result.isAI;
        
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'ai-detector-overlay';
        overlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: ${isAI ? 'rgba(220, 53, 69, 0.8)' : 'rgba(40, 167, 69, 0.8)'};
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            border-radius: 4px;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
            z-index: 1000;
        `;
        
        overlay.innerHTML = `
            <div style="text-align: center;">
                <div style="font-size: 16px; margin-bottom: 4px;">
                    ${isAI ? '🤖' : '✅'}
                </div>
                <div>${isAI ? 'AI' : 'Real'}</div>
                <div style="font-size: 10px;">${confidence}%</div>
            </div>
        `;

        // Make image container relative if needed
        const container = img.parentElement;
        if (container && getComputedStyle(container).position === 'static') {
            container.style.position = 'relative';
        }

        // Add overlay
        container.appendChild(overlay);

        // Show overlay on hover
        container.addEventListener('mouseenter', () => {
            overlay.style.opacity = '1';
        });

        container.addEventListener('mouseleave', () => {
            overlay.style.opacity = '0';
        });

        // Add border to image
        img.style.border = `2px solid ${isAI ? '#dc3545' : '#28a745'}`;
        img.style.borderRadius = '4px';
    }
}

// Initialize content script
new AIImageDetectorContent();

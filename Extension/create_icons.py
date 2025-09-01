#!/usr/bin/env python3
"""
Create simple icon files for the Chrome extension
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    """Create a simple icon with the specified size"""
    # Create a new image with a gradient background
    img = Image.new('RGBA', (size, size), (102, 126, 234, 255))
    draw = ImageDraw.Draw(img)
    
    # Add a simple AI symbol (robot face)
    # Draw a circle for the head
    margin = size // 4
    draw.ellipse([margin, margin, size - margin, size - margin], 
                 fill=(255, 255, 255, 200), outline=(255, 255, 255, 255), width=2)
    
    # Draw eyes
    eye_size = size // 8
    left_eye_x = size // 3
    right_eye_x = 2 * size // 3
    eye_y = size // 2 - eye_size // 2
    
    draw.ellipse([left_eye_x - eye_size//2, eye_y, left_eye_x + eye_size//2, eye_y + eye_size], 
                 fill=(102, 126, 234, 255))
    draw.ellipse([right_eye_x - eye_size//2, eye_y, right_eye_x + eye_size//2, eye_y + eye_size], 
                 fill=(102, 126, 234, 255))
    
    # Draw a simple mouth
    mouth_width = size // 4
    mouth_height = size // 12
    mouth_x = size // 2 - mouth_width // 2
    mouth_y = 3 * size // 4 - mouth_height // 2
    
    draw.rectangle([mouth_x, mouth_y, mouth_x + mouth_width, mouth_y + mouth_height], 
                   fill=(102, 126, 234, 255))
    
    # Save the image
    img.save(filename)
    print(f"Created {filename}")

def main():
    """Create all required icon files"""
    # Ensure icons directory exists
    os.makedirs("icons", exist_ok=True)
    
    # Create icons for different sizes
    create_icon(16, "icons/icon16.png")
    create_icon(48, "icons/icon48.png")
    create_icon(128, "icons/icon128.png")
    
    print("✅ All icon files created successfully!")

if __name__ == "__main__":
    main()

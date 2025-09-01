#!/usr/bin/env python3
"""
Script to update the Chrome extension background.js with Colab URL
"""

import re
import sys

def update_backend_url(url):
    """Update the backend URL in background.js"""
    try:
        # Read the current background.js
        with open('background.js', 'r') as f:
            content = f.read()
        
        # Replace the URL
        pattern = r"this\.pythonBackendUrl = 'http://localhost:5000';"
        replacement = f"this.pythonBackendUrl = '{url}';"
        
        if pattern in content:
            new_content = content.replace(pattern, replacement)
            
            # Write back to file
            with open('background.js', 'w') as f:
                f.write(new_content)
            
            print(f"✅ Successfully updated background.js with URL: {url}")
            return True
        else:
            print("❌ Could not find the URL pattern in background.js")
            print("Please manually update the URL in background.js:")
            print(f"Change: this.pythonBackendUrl = 'http://localhost:5000';")
            print(f"To: this.pythonBackendUrl = '{url}';")
            return False
            
    except Exception as e:
        print(f"❌ Error updating background.js: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python update_colab_url.py <ngrok_url>")
        print("Example: python update_colab_url.py https://abc123.ngrok.io")
        return
    
    url = sys.argv[1]
    
    # Validate URL format
    if not url.startswith('https://'):
        print("❌ URL must start with 'https://'")
        return
    
    print(f"🔄 Updating background.js with Colab URL: {url}")
    update_backend_url(url)
    
    print("\n📋 Next steps:")
    print("1. Reload the Chrome extension in chrome://extensions/")
    print("2. Test the extension on a webpage with images")
    print("3. The extension should now connect to your Colab backend!")

if __name__ == "__main__":
    main()

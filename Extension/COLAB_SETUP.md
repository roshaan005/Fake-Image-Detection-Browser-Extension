# Google Colab Setup Instructions

## How to Run the Backend in Google Colab

Yes, you can absolutely run the backend in Google Colab! This is actually a great idea because:

1. **Free GPU access** - Colab provides free GPU/TPU for faster training
2. **Pre-installed dependencies** - Most ML libraries are already available
3. **No local setup required** - Everything runs in the cloud
4. **Public URL** - ngrok creates a public URL for your Chrome extension

## Step-by-Step Instructions

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Create a new notebook

### 2. Install Dependencies
```python
!pip install tensorflow scikit-learn opencv-python Pillow flask flask-cors numpy matplotlib seaborn pandas requests python-dotenv
!pip install pyngrok
```

### 3. Copy the Backend Code
Copy the entire content of `ai_detector.py` into a Colab cell and run it.

### 4. Start the Server
The code will automatically:
- Install all dependencies
- Start an ngrok tunnel
- Create a public URL
- Start the Flask server

### 5. Update Chrome Extension
When the Colab notebook runs, it will output a public URL like:
```
Public URL: https://abc123.ngrok.io
```

Copy this URL and update your `background.js` file:
```javascript
this.pythonBackendUrl = 'https://abc123.ngrok.io';  // Replace with your URL
```

### 6. Load the Extension
- Load the Chrome extension as normal
- The extension will now connect to your Colab backend

## Advantages of Colab Setup

✅ **No local Python installation needed**
✅ **Free GPU acceleration**
✅ **Always-on server** (while Colab is running)
✅ **Public URL** accessible from anywhere
✅ **Pre-installed ML libraries**

## Important Notes

⚠️ **Colab Limitations:**
- Runtime disconnects after 12 hours of inactivity
- Limited memory (but sufficient for this project)
- Need to restart if disconnected

⚠️ **Security:**
- The ngrok URL is public - anyone can access it
- For production use, add authentication

## Alternative: Local Setup

If you prefer to run locally:
1. Install Python dependencies: `pip install -r requirements.txt`
2. Run: `python start.py`
3. Extension connects to `http://localhost:5000`

## Testing the Setup

After setting up, test with:
```bash
python test_installation.py
```

This will verify that your Chrome extension can connect to the Colab backend.

# Plant Disease Classification Web Application

## Overview
This web application uses a deep learning model (AlexNet) to classify plant leaf diseases. The model can identify 38 different classes of plant diseases across various crops, helping with early detection and diagnosis of plant health issues.

## Contents
- `app.py` - Flask web server application
- `requirements.txt` - Python dependencies
- `AlexNetModel.hdf5` - Pre-trained deep learning model
- `templates/` - Web interface templates (auto-created on first run)
- Sample leaf images for testing

## Prerequisites
- Python 3.6 or higher
- pip package manager
- 4GB+ RAM recommended
- ~500MB disk space for dependencies and the model

## Installation Instructions

### 1. Environment Setup
optional step
Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```

The installation might take a few minutes depending on your internet connection and computer performance.

### 3. Launch the Application
Run the Flask server:

```bash
python app.py
```

You should see output indicating that the model is loading and the server is starting.

### 4. Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5001
```

## Using the Application

1. From the web interface, click the "Choose File" button
2. Select an image of a plant leaf (JPG, JPEG, or PNG format)
3. Click the "Analyze Disease" button
4. View the diagnosis results showing the detected disease and confidence level

## Sample Images
The folder includes some sample images you can use to test the application:
- `AppleCedarRust1.JPG` - Apple leaf with Cedar Rust disease
- `AppleCedarRust4.JPG` - Another sample of Apple Cedar Rust
- `TomatoHealthy3.JPG` - Healthy tomato leaf
- `TomatoYellowCurlVirus5.JPG` - Tomato Yellow Curl Virus

## Troubleshooting

- **Model not found error**: Ensure the `AlexNetModel.hdf5` file is in the same directory as `app.py`
- **Dependency errors**: Check that you're using the correct versions specified in requirements.txt
- **Memory errors**: The model requires significant RAM. Close other applications if needed
- **Slow predictions**: The first prediction may take longer as the model loads into memory

## Notes

- The application runs locally and is intended for educational/research purposes
- For production use, additional security measures would be required
- The model works best with clear, well-lit images of individual leaves

## Supported Plant Diseases

The model can identify diseases across various crops including:
- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Strawberry
- Tomato

And many common diseases such as:
- Early Blight
- Late Blight
- Leaf Spot
- Powdery Mildew
- Rust
- Black Rot
- Bacterial Spot
- And healthy plant identification
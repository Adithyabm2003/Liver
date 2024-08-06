import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import cv2

# Initialize the Flask application
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = tf.keras.models.load_model('liver_tumor_detection_mod.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to extract HOG features
def extract_hog_features(image):
    if image.ndim == 3:
        # RGB image
        image = rgb2gray(image)
    # Resize image to consistent shape if needed
    if image.shape != (60, 120):
        image = resize(image, (130, 130))
    # Calculate HOG features
    hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
    return hog_features

# Function to preprocess a new image and make predictions
def predict_tumor(image_path):
    # Load the image
    image = imread(image_path)
    
    # Drop alpha channel if present
    if image.shape[-1] == 4:
        image = image[..., :3]  # Keep only RGB channels, drop alpha
    
    # Extract HOG features
    hog_features = extract_hog_features(image)
    
    # Reshape HOG features for CNN input
    hog_features = hog_features.reshape(1, 60, 120, 1)  # Reshape to (1, 60, 120, 1)
    
    # Make predictions
    predictions = model.predict(hog_features)
    
    # Get the predicted class (0 or 1 for binary classification)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class == 0:
        return "No tumor detected."
    else:
        return "Tumor detected."

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict_tumor(file_path)
            return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)

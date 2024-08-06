import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

# Load the trained model
model = tf.keras.models.load_model('liver_tumor_detection_mod.h5')

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
        print("No tumor detected.")
    else:
        print("Tumor detected.")



# Example usage:
image_path1 = "static/timg1.jpg"
image_path2 = "static/img2.jpg"
image_path3 = "static/img3.jpg"

predict_tumor(image_path1)

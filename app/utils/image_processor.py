import numpy as np
import tensorflow as tf
from PIL import Image

def prepare_image(image: Image.Image, target_size=(224, 224)):
    """
    Prepare an image for model prediction
    
    Args:
        image (PIL.Image.Image): Input image
        target_size (tuple): Desired image size for model input
    
    Returns:
        numpy.ndarray: Preprocessed image array ready for prediction
    """
    # Resize image
    resized_image = image.resize(target_size)
    
    # Convert to RGB if needed
    if resized_image.mode != 'RGB':
        resized_image = resized_image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(resized_image)
    
    # Expand dimensions for model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess using MobileNetV2 preprocessing
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return img_array

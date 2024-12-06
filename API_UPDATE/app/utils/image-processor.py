import numpy as np
import tensorflow as tf
from PIL import Image

def prepare_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Prepare image for model prediction
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Resize dimensions for the image
    
    Returns:
        np.ndarray: Preprocessed image array
    """
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_recommended_hairstyles(face_shape: str) -> list[str]:
    """
    Get recommended hairstyles based on face shape
    
    Args:
        face_shape (str): Predicted face shape
    
    Returns:
        list[str]: Recommended hairstyles
    """
    hairstyle_recommendations = {
        "Oval": [
            "Side-Swept Bangs",
            "Long Layers",
            "Pixie Cut",
            "Bob"
        ],
        "Round": [
            "Long Layered Cut",
            "Side-Swept Bangs",
            "High Ponytail",
            "Asymmetrical Bob"
        ],
        "Square": [
            "Soft Waves",
            "Long Layers with Soft Texture",
            "Shoulder-Length Cut",
            "Layered Pixie"
        ],
        "Heart": [
            "Chin-Length Bob",
            "Side-Swept Bangs",
            "Soft Waves",
            "Textured Lob"
        ],
        "Diamond": [
            "Chin-Length Bob",
            "Pixie Cut",
            "Side-Swept Bangs",
            "Medium Length with Layers"
        ]
    }
    
    return hairstyle_recommendations.get(face_shape, [])

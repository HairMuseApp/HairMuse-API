from pydantic import BaseModel, Field
from typing import List

class PredictionResponse(BaseModel):
    """
    Schema for prediction response
    
    Attributes:
        face_shape (str): Predicted face shape
        confidence (float): Confidence of the prediction in percentage
    """
    uploaded_image: str
    face_shape: str
    confidence: float
    description: str
    tips: List[str]
    recommendations: List[str]
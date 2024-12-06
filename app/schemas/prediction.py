from pydantic import BaseModel, Field
<<<<<<< HEAD
from fastapi import UploadFile
from typing import List, Dict
=======
>>>>>>> 35b665c7f7672268308905183867168de0db2fce

class PredictionResponse(BaseModel):
    """
    Schema for prediction response
    
    Attributes:
        face_shape (str): Predicted face shape
        confidence (float): Confidence of the prediction in percentage
    """
<<<<<<< HEAD
    uploaded_image: str
    face_shape: str
    confidence: float
    description: str
    tips: List[str]
    recommendations: List[Dict[str, str]]
    
class PredictionRequest(BaseModel):
    file: UploadFile
    gender: str = Field(..., pattern="^(male|female)$")
=======
    face_shape: str = Field(..., description="Predicted face shape")
    confidence: float = Field(..., description="Confidence of prediction", gt=0, le=100)
>>>>>>> 35b665c7f7672268308905183867168de0db2fce

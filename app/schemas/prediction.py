from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """
    Schema for prediction response
    
    Attributes:
        face_shape (str): Predicted face shape
        confidence (float): Confidence of the prediction in percentage
    """
    face_shape: str = Field(..., description="Predicted face shape")
    confidence: float = Field(..., description="Confidence of prediction", gt=0, le=100)

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    Schema for face shape prediction response
    """
    face_shape: str
    confidence: float
    recommended_hairstyles: list[str] = []

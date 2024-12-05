from pydantic import BaseModel, HttpUrl
from typing import List

class HairstyleRecommendation(BaseModel):
    """
    Skema untuk rekomendasi gaya rambut
    """
    filename: str
    image_url: HttpUrl

class PredictionResponse(BaseModel):
    """
    Skema respons prediksi bentuk wajah
    """
    face_shape: str
    confidence: float
    recommended_hairstyles: List[HairstyleRecommendation] = []

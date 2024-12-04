import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from PIL import Image
import io
import json

from app.schemas.prediction import PredictionResponse
from app.utils.image_processor import prepare_image

# Create FastAPI app instance
app = FastAPI(
    title="Face Shape Prediction API",
    description="API for predicting face shapes from uploaded images"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define class indices
current_dir = os.path.dirname(__file__) 
file_path = os.path.join(current_dir, 'class_indices.json')

with open(file_path, 'r') as f:
    loaded_class_indices = json.load(f)
    
CLASS_NAMES = {v: k for k, v in loaded_class_indices.items()}

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
async def predict_face_shape(file: UploadFile = File(...)):
    """
    Predict face shape from an uploaded image
    
    - Accepts a single image file
    - Returns predicted face shape and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Open image using Pillow
        image = Image.open(io.BytesIO(contents))
        
        # Prepare image for prediction
        img_array = prepare_image(image)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Get class name and confidence
        class_name = CLASS_NAMES[predicted_class[0]]
        confidence = float(np.max(predictions) * 100)
        
        return {
            "face_shape": class_name,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Simple health check endpoint
    """
    return {"message": "Face Shape Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

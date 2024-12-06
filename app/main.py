import os
import numpy as np
<<<<<<< HEAD
import tensorflow as tf
from PIL import Image
import io
import json
import random
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.prediction import PredictionResponse, PredictionRequest
=======
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

import tensorflow as tf
from PIL import Image
import io

from app.schemas.prediction import PredictionResponse
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
from app.utils.image_processor import prepare_image

# Create FastAPI app instance
app = FastAPI(
    title="Face Shape Prediction API",
<<<<<<< HEAD
    description="API for predicting face shapes from uploaded images"
=======
    description="API for predicting face shapes from uploaded images",
    version="1.0.0"
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

<<<<<<< HEAD
# Define class indices
=======
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
current_dir = os.path.dirname(__file__) 
file_path = os.path.join(current_dir, 'class_indices.json')

with open(file_path, 'r') as f:
    loaded_class_indices = json.load(f)
<<<<<<< HEAD
    
=======

# Convert to class names dictionary
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
CLASS_NAMES = {v: k for k, v in loaded_class_indices.items()}

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
<<<<<<< HEAD
async def predict_face_shape(file: UploadFile = File(...), gender: str = "female"):
=======
async def predict_face_shape(file: UploadFile = File(...)):
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
    """
    Predict face shape from an uploaded image
    
    - Accepts a single image file
    - Returns predicted face shape and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
<<<<<<< HEAD
    # Validate gender
    if gender not in ["male", "female"]:
        gender = 'female'
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")  
    
=======
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
    try:
        # Read image file
        contents = await file.read()
        
<<<<<<< HEAD
       # Simpan gambar yang di-upload ke folder lokal
        upload_folder = 'static/images'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, "uploaded_image.jpg")
        
        with open(file_path, "wb") as f:
            f.write(contents)

        uploaded_image_url = f"/static/images/uploaded_image.jpg" 
        
=======
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
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
<<<<<<< HEAD

        # Load details from JSON
        details_path = os.path.join(current_dir, 'face_shape_details.json')
        
        with open(details_path, 'r') as f:
            face_shape_details = json.load(f)
        
        details = face_shape_details.get(class_name, {})
        description = details.get("description", "No description available")
        tips = details.get("tips", ["No tips available"])
        
        # Get hairstyle recommendations (3 random images)
        hairstyle_folder = os.path.join(current_dir, 'hairstyle_database', gender, class_name)
        hairstyle_images = []
        
        try:
            hairstyle_files = os.listdir(hairstyle_folder)
            hairstyle_images = random.sample(hairstyle_files, 3)  # Take 3 random images
        except Exception as e:
            print(f"Error loading hairstyle images: {e}")
        
        # Mengambil rekomendasi potongan rambut dengan URL (bukan base64)
        hairstyle_images_urls = []
        for img_file in hairstyle_images:
            img_url = f"/static/hairstyle/{gender}/{class_name}/{img_file}"  # URL lokal
            hairstyle_images_urls.append({
                "filename": img_file,
                "image": img_url
            })
        
        # After sending the response, remove the temporary uploaded image file
        os.remove(file_path)
        return {
            "uploaded_image": uploaded_image_url,
            "face_shape": class_name,
            "confidence": confidence,
            "description": description,
            "tips": tips,
            "recommendations": hairstyle_images_urls
        }
        
=======
        
        return {
            "face_shape": class_name,
            "confidence": confidence
        }
    
>>>>>>> 35b665c7f7672268308905183867168de0db2fce
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

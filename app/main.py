import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json
import random
from enum import Enum


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas.prediction import PredictionResponse, PredictionRequest
from app.utils.image_processor import prepare_image


class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"

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


"""
Kode untuk upload ke Google cloud storage
"""
# Set up the static files path
static_folder = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_folder, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_folder), name="static")


# Load model
model_local_path = os.path.join("app", "models", "model.keras")
model = tf.keras.models.load_model(model_local_path, compile=False)

# Define class indices
current_dir = os.path.dirname(__file__) 
file_path = os.path.join(current_dir, 'utils', 'class_indices.json')

with open(file_path, 'r') as f:
    loaded_class_indices = json.load(f)
    
CLASS_NAMES = {v: k for k, v in loaded_class_indices.items()}


@app.post("/predict", response_model=PredictionResponse)
async def predict_face_shape(
    file: UploadFile = File(...), 
    gender: GenderEnum = GenderEnum.FEMALE  # Default value
    ):
    """
    Predict face shape from an uploaded image
    
    - Accepts a single image file
    - Returns predicted face shape and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate gender
    if gender not in ["male", "female"]:
        gender = 'female'
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")  
    
    try:
        # Read image file
        contents = await file.read()
        
       # Save uploaded image to a local folder
        upload_folder = os.path.join(current_dir, 'static', 'images')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, "uploaded_image.jpg")
        
        with open(file_path, "wb") as f:
            f.write(contents)

        uploaded_image_url = f"/static/images/uploaded_image.jpg" 
        
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

        # Load details from JSON
        details_path = os.path.join(current_dir, 'utils', 'face_shape_details.json')
        
        with open(details_path, 'r') as f:
            face_shape_details = json.load(f)
        
        details = face_shape_details.get(class_name, {})
        description = details.get("description", "No description available")
        tips = details.get("tips", ["No tips available"])

        
        # # Get hairstyle recommendations (3 random images)
        hairstyle_folder = os.path.join(current_dir, 'hairstyle_database', gender.value, class_name)
        
        # Ensure the folder exists
        if not os.path.exists(hairstyle_folder):
            raise HTTPException(status_code=404, detail=f"Hairstyle folder for {class_name} not found")

        # Get list of hairstyle images in the folder
        all_images = [f for f in os.listdir(hairstyle_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # If there are less than 3 images, return all available images
        num_images = min(3, len(all_images))
        
        # Select random images
        selected_images = random.sample(all_images, num_images)

        # Prepare the URLs for the selected images
        hairstyle_images_urls = []
        for img_file in selected_images:
            img_url = f"/hairstyle_database/{gender.value}/{class_name}/{img_file}"
            hairstyle_images_urls.append({
                "filename": os.path.basename(img_file),
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/hairstyles/{gender}/{face_shape}")
async def get_hairstyles(gender: str, face_shape: str):
    """
    Retrieve all hairstyle images based on gender and face shape.
    
    - gender: "male" or "female"
    - face_shape: one of ["heart", "oblong", "oval", "round", "square"]
    """
    # Validate gender
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
    
    # Validate face shape
    valid_face_shapes = ["heart", "oblong", "oval", "round", "square"]
    if face_shape not in valid_face_shapes:
        raise HTTPException(status_code=400, detail=f"Invalid face shape. Must be one of {valid_face_shapes}")
    
    # Define the folder path
    current_dir = os.path.dirname(__file__)
    hairstyle_folder = os.path.join(current_dir, 'hairstyle_database', gender, face_shape)
    
    # Check if the folder exists
    if not os.path.exists(hairstyle_folder):
        raise HTTPException(status_code=404, detail=f"No images found for {gender} {face_shape}")
    
    # List all images in the folder
    image_files = [f for f in os.listdir(hairstyle_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise HTTPException(status_code=404, detail=f"No images found for {gender} {face_shape}")
    
    # Generate URLs for all images
    image_urls = [{"filename": img_file, "url": f"/hairstyle_database/{gender}/{face_shape}/{img_file}"} for img_file in image_files]
    
    # Include face shape in the response
    return {
        "face_shape": face_shape,
        "images": image_urls
    }

@app.get("/")
async def root():
    """
    Simple health check endpoint
    """
    return {"message": "Face Shape Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

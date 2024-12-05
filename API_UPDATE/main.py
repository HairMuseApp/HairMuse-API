import os
import urllib.parse 
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from PIL import Image
import io

from app.schemas.prediction import PredictionResponse
from app.utils.image_processor import prepare_image, HairstyleRecommender

# Inisialisasi FastAPI
app = FastAPI(
    title="Face Shape Prediction API",
    description="API untuk prediksi bentuk wajah dan rekomendasi gaya rambut",
    version="1.0.0"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Menyajikan file statis dari folder local_hairstyles
hairstyle_folder = os.path.join(os.path.dirname(__file__), "local_hairstyles", "Database Hairstyle")
app.mount("/images", StaticFiles(directory=hairstyle_folder), name="images")


# Inisialisasi recommender
hairstyle_recommender = HairstyleRecommender()

# Load class indices
file_path = os.path.join(os.path.dirname(__file__), 'class_indices.json')

with open(file_path, 'r') as f:
    loaded_class_indices = json.load(f)

# Konversi indeks kelas
CLASS_NAMES = {v: k for k, v in loaded_class_indices.items()}

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'app', 'models', 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
async def predict_face_shape(file: UploadFile = File(...)):
    """
    Prediksi bentuk wajah dan dapatkan rekomendasi gaya rambut
    """
    # Validasi tipe file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    
    try:
        # Baca file gambar
        contents = await file.read()
        
        # Buka gambar dengan Pillow
        image = Image.open(io.BytesIO(contents))
        
        # Persiapkan gambar untuk prediksi
        img_array = prepare_image(image)
        
        # Lakukan prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Dapatkan nama kelas dan confidence
        class_name = CLASS_NAMES[predicted_class[0]]
        confidence = float(np.max(predictions) * 100)
        
        # Dapatkan rekomendasi gaya rambut dengan gambar
        recommended_hairstyles = hairstyle_recommender.get_hairstyle_recommendations(class_name)
        
        # Menambahkan URL gambar dengan memastikan encoding nama file
        for hairstyle in recommended_hairstyles:
            # Encode filename untuk memastikan tidak ada spasi atau karakter tidak valid dalam URL
            encoded_filename = urllib.parse.quote(hairstyle['filename'].replace(' ', '_'))  # Ganti spasi dengan _
            hairstyle["image_url"] = f"http://127.0.0.1:8000/images/{class_name.lower()}/{encoded_filename}.jpg"
        
        return PredictionResponse(
            face_shape=class_name,
            confidence=confidence,
            recommended_hairstyles=recommended_hairstyles
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Endpoint health check
    """
    return {"message": "Face Shape Prediction API berjalan"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

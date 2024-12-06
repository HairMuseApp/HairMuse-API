import os
# from google.cloud import storage
import numpy as np
import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

class HairstyleRecommender:
    def __init__(self, base_dir='local_hairstyles/Database Hairstyle'):
        """
        Inisialisasi direktori lokal untuk menyimpan gambar gaya rambut.

        Args:
            base_dir (str): Path ke direktori lokal yang berisi gambar.
        """
        self.base_dir = base_dir
        
        
    """
    INI UNTUK GOOGLE CLOUD STORAGE
    """
    # def __init__(self, bucket_name='hairstyle-recommendations'):
    #     """
    #     Inisialisasi Google Cloud Storage client
        
    #     Args:
    #         bucket_name (str): Nama bucket di Google Cloud Storage
    #     """
    #     self.storage_client = storage.Client()
    #     self.bucket_name = bucket_name
    #     self.bucket = self.storage_client.bucket(bucket_name)

    def get_hairstyle_recommendations(self, face_shape: str) -> list:
        """
        Dapatkan rekomendasi gaya rambut dengan detail gambar
        
        Args:
            face_shape (str): Bentuk wajah yang diprediksi
        
        Returns:
            list[dict]: Daftar rekomendasi gaya rambut dengan detail
        """
        
        # hairstyle_mappings = {
        #     "Oval": [
        #         "side_swept_bangs.jpg",
        #         "long_layers.jpg",
        #         "pixie_cut.jpg"
        #     ],
        #     "Round": [
        #         "long_layered_cut.jpg", 
        #         "side_swept_bangs.jpg",
        #         "asymmetrical_bob.jpg"
        #     ],
        #     "Square": [
        #         "soft_waves.jpg",
        #         "shoulder_length.jpg",
        #         "layered_pixie.jpg"
        #     ],
        #     "Heart": [
        #         "chin_length_bob.jpg",
        #         "side_swept_bangs.jpg",
        #         "textured_lob.jpg"
        #     ],
        #     "Diamond": [
        #         "chin_length_bob.jpg",
        #         "pixie_cut.jpg",
        #         "medium_layers.jpg"
        #     ]
        # }
        
        # # Dapatkan daftar file gambar untuk bentuk wajah
        # hairstyle_files = hairstyle_mappings.get(face_shape, [])
        
        # Buat list untuk menyimpan detail gambar

         # Path ke folder yang sesuai dengan face_shape
        folder_path = os.path.join(self.base_dir, face_shape.lower())

        # Periksa apakah folder ada
        if not os.path.exists(folder_path):
            return []

        # Dapatkan daftar file gambar dalam folder
        hairstyle_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

        hairstyle_files = hairstyle_files[:5]
        
        hairstyle_details = []
        
        base_url = "http://127.0.0.1:8000/images" 
        
        for filename in hairstyle_files:
            image_url = f"{base_url}/{face_shape.lower()}/{filename}"
            hairstyle_details.append({
                "filename": filename.split('.')[0].replace('_', ' ').title(),
                "image_path": image_url  # Path lokal gambar
            })
            
        # for filename in hairstyle_files:
        #     # Buat blob (referensi objek) dari Google Cloud Storage
        #     blob = self.bucket.blob(f"{face_shape.lower()}/{filename}")
            
        #     # Dapatkan URL publik gambar
        #     image_url = blob.public_url
            
        #     hairstyle_details.append({
        #         "filename": filename.split('.')[0].replace('_', ' ').title(),
        #         "image_url": image_url
        #     })
        
        return hairstyle_details

# Fungsi untuk menyiapkan gambar (tetap sama seperti sebelumnya)
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
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessing menggunakan MobileNetV2
    img_array = preprocess_input(img_array)
    
    return img_array

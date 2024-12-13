# Face Shape Prediction API

## Project Description
This is a FastAPI-based machine learning service for predicting face shapes from uploaded images using a pre-trained TensorFlow model.

## Prerequisites
- Python 3.9+
- pip

## Installation

1. Clone the repository
```bash
git clone https://github.com/HairMuseApp/HairMuse-API.git
cd face-shape-prediction-api
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Place your trained model
- Put your `model.keras` in the `app/models/` directory

## Running the API

```bash
uvicorn app.main:app --reload

#atau 

python -m app.main 
```

## Google Cloud Architecture
<img src="Cloud Architecture.png" alt="HairMuse Logo" width="990" height="650">

## BASE URL: https://hairmuseimg2-325820985735.asia-southeast2.run.app/
- `/` [GET]: Health check endpoint
- `/hairstyles/{gender}/{face_shape}` [GET]: Getting Images from hairstyle_database
- `/predict` [POST]: Upload an image to get face shape prediction

## Usage Example
You can use tools like curl, Postman, or write a frontend to interact with the API:

```python
import requests

url = "http://localhost:8080/predict"
with open("test_image.jpg", "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)
    print(response.json())
```

## Model Details
- Predicts face shapes: heart, oblong, oval, round, square
- Uses ResNet50 preprocessing
- Input image size: 224x224 pixels

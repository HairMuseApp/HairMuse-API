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
- Put your `best_model.keras` in the `app/models/` directory

## Running the API

```bash
uvicorn app.main:app --reload
```

## API Endpoints
- `/predict` [POST]: Upload an image to get face shape prediction
- `/` [GET]: Health check endpoint

## Usage Example
You can use tools like curl, Postman, or write a frontend to interact with the API:

```python
import requests

url = "http://localhost:8000/predict"
with open("test_image.jpg", "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)
    print(response.json())
```

## Model Details
- Predicts face shapes: heart, oblong, oval, round, square
- Uses MobileNetV2 preprocessing
- Input image size: 224x224 pixels

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

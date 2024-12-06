# Face Shape Prediction API

## Project Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model
- Place your trained model at `app/models/best_model.keras`
- Create `class_indices.json` with class mappings

### 4. Run the Application
```bash
uvicorn main:app --reload
```

## API Endpoints
- `/predict`: Upload image for face shape prediction
- `/`: Health check endpoint

## Model Requirements
- Input image size: 224x224 pixels
- Supported formats: JPEG, PNG

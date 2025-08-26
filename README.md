# AdversarialAttack

A modular demo of adversarial attacks on pretrained computer vision models using FastAPI (backend) and Streamlit (frontend).

## Structure
- `backend/`: FastAPI app, PyTorch models, attack implementations
- `frontend/`: Streamlit UI
- `download_models.py`: Script to pre-download all models

## Setup

### 1. Install Dependencies
To install all dependencies, run:
```sh
uv sync
```

### 2. Download Models (Recommended)
Pre-download all pretrained models to avoid delays during runtime:
```sh
python download_models.py
```

This script will download:
- ResNet18
- EfficientNet_B0  
- MobileNetV2

The models will be cached locally (~100-200MB total) for faster loading.

### 3. Start the Services

#### Backend
Run FastAPI server:
```sh
uvicorn backend.main:app --reload
```

#### Frontend
Run Streamlit app:
```sh
streamlit run frontend/app.py
```

### 4. Access the Demo
- Open your browser to `http://localhost:8501` for the Streamlit UI
- The FastAPI backend will be running on `http://localhost:8000`
- API documentation available at `http://localhost:8000/docs`

## Usage

1. **Upload an image** using the file uploader
2. **Choose a model** from the sidebar (ResNet18, EfficientNet_B0, MobileNetV2)
3. **Select an attack type** and adjust parameters in the sidebar
4. **Review the attack configuration** displayed below the upload
5. **Click "Launch Attack"** to generate adversarial examples
6. **Compare results** side-by-side to see how the attack affects predictions

## Features
- Choose pretrained model (ResNet18, EfficientNet_B0, MobileNetV2)
- Choose attack type (FGSM, PGD, Gaussian Blur, Salt-and-Pepper Noise, Adversarial Patch)
- Adjust attack parameters dynamically
- View original and adversarial predictions side-by-side
- Detailed probability bar chart comparisons
- Attack success metrics and analysis
- Modern, responsive UI with sidebar controls

## Available Attacks

### Adversarial Attacks
- **FGSM (Fast Gradient Sign Method)**: Single-step gradient-based attack
- **PGD (Projected Gradient Descent)**: Multi-step iterative attack

### Image Corruptions  
- **Gaussian Blur**: Simple image blurring with configurable kernel size
- **Salt & Pepper Noise**: Random pixel corruption with configurable noise level
- **Adversarial Patch**: Overlay attack with a bright square patch

## Technical Details

### Backend (FastAPI)
- RESTful API with `/predict/` and `/attack/` endpoints
- Modular attack implementations in `backend/attacks/`
- Comprehensive error handling and logging
- Input validation and sanitization

### Frontend (Streamlit)
- Modern UI with sidebar controls and responsive layout
- Real-time parameter adjustment based on attack type
- Progress indicators and status feedback
- Interactive charts and metrics display

## Notes
- All attacks implemented in `backend/attacks/`
- Models and utils in `backend/models.py` and `backend/utils.py`
- FastAPI and Streamlit run as separate services
- Uses PyTorch and torchvision for models and attacks
- Pre-downloading models recommended for better performance

# AdversarialAttack

A modular demo of adversarial attacks on pretrained computer vision models using FastAPI (backend) and Streamlit (frontend).

## Structure
- `backend/`: FastAPI app, PyTorch models, attack implementations
- `frontend/`: Streamlit UI

## Setup

To install all dependencies, run:
```sh
uv sync
```

### Backend
Run FastAPI server:
```sh
uvicorn backend.main:app --reload
```

### Frontend
Run Streamlit app:
```sh
streamlit run frontend/app.py
```

## Features
- Choose pretrained model (ResNet18, EfficientNet_B0, MobileNetV2)
- Choose attack type (FGSM, PGD, Gaussian Blur, Salt-and-Pepper Noise, Adversarial Patch)
- Set attack parameters
- View original and adversarial predictions side-by-side
- Probability bar chart comparison

## Notes
- All attacks implemented in `backend/attacks/`
- Models and utils in `backend/models.py` and `backend/utils.py`
- FastAPI and Streamlit run as separate services
- Uses PyTorch and torchvision

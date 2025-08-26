from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .models import get_model, get_imagenet_labels
from .utils import preprocess_image, get_top5_predictions, image_to_base64
from .attacks import fgsm, pgd, blur, sp_noise, patch
import io
import torch
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adversarial Attacks API",
    description="API for demonstrating adversarial attacks on computer vision models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Adversarial Attacks API", "status": "running"}

@app.post("/predict/")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    try:
        if model_name not in ["ResNet18", "EfficientNet_B0", "MobileNetV2"]:
            raise HTTPException(status_code=400, detail="Invalid model name")
            
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model = get_model(model_name)
        input_tensor = preprocess_image(image)
        top5 = get_top5_predictions(model, input_tensor)
        
        logger.info(f"Prediction successful for model: {model_name}")
        return JSONResponse(top5)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/attack/")
async def attack(
    model_name: str = Form(...),
    attack_type: str = Form(...),
    epsilon: float = Form(0.03),
    steps: int = Form(10),
    kernel_size: int = Form(3),
    noise_level: float = Form(0.05),
    file: UploadFile = File(...)
):
    try:
        if model_name not in ["ResNet18", "EfficientNet_B0", "MobileNetV2"]:
            raise HTTPException(status_code=400, detail="Invalid model name")
            
        if attack_type not in ["FGSM", "PGD", "GaussianBlur", "SaltPepper", "Patch"]:
            raise HTTPException(status_code=400, detail="Invalid attack type")
            
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model = get_model(model_name)
        input_tensor = preprocess_image(image)
        orig_preds = get_top5_predictions(model, input_tensor)
        
        # Generate adversarial example
        if attack_type == "FGSM":
            adv_tensor = fgsm(model, input_tensor, epsilon)
        elif attack_type == "PGD":
            adv_tensor = pgd(model, input_tensor, epsilon, steps)
        elif attack_type == "GaussianBlur":
            adv_tensor = blur(input_tensor, kernel_size)
        elif attack_type == "SaltPepper":
            adv_tensor = sp_noise(input_tensor, noise_level)
        elif attack_type == "Patch":
            adv_tensor = patch(input_tensor)
        
        # Convert adversarial tensor back to image
        adv_numpy = adv_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        adv_numpy = (adv_numpy * 255).clip(0, 255).astype('uint8')
        adv_image = Image.fromarray(adv_numpy)
        
        adv_preds = get_top5_predictions(model, adv_tensor)
        adv_image_b64 = image_to_base64(adv_image)
        
        logger.info(f"Attack successful: {attack_type} on {model_name}")
        return JSONResponse({
            "original": orig_preds,
            "adversarial": adv_preds,
            "adv_image": adv_image_b64,
            "attack_info": {
                "type": attack_type,
                "model": model_name,
                "parameters": {
                    "epsilon": epsilon if attack_type in ["FGSM", "PGD"] else None,
                    "steps": steps if attack_type == "PGD" else None,
                    "kernel_size": kernel_size if attack_type == "GaussianBlur" else None,
                    "noise_level": noise_level if attack_type == "SaltPepper" else None
                }
            }
        })
    except Exception as e:
        logger.error(f"Attack error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Attack failed: {str(e)}")

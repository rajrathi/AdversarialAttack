from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .models import get_model, get_imagenet_labels
from .utils import preprocess_image, get_top5_predictions, image_to_base64
from .attacks import fgsm, pgd, blur, sp_noise, patch
import io
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = get_model(model_name)
    input_tensor = preprocess_image(image)
    top5 = get_top5_predictions(model, input_tensor)
    return JSONResponse(top5)

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
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = get_model(model_name)
    input_tensor = preprocess_image(image)
    orig_preds = get_top5_predictions(model, input_tensor)
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
    else:
        return JSONResponse({"error": "Unknown attack type"}, status_code=400)
    adv_image = Image.fromarray((adv_tensor.squeeze().permute(1,2,0).cpu().numpy()*255).astype('uint8'))
    adv_preds = get_top5_predictions(model, adv_tensor)
    adv_image_b64 = image_to_base64(adv_image)
    return JSONResponse({
        "original": orig_preds,
        "adversarial": adv_preds,
        "adv_image": adv_image_b64
    })

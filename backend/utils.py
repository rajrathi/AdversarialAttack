import torch
from torchvision import transforms
import base64
from io import BytesIO
from PIL import Image
from .models import get_imagenet_labels

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def get_top5_predictions(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)
    labels = get_imagenet_labels()
    return [{"class": labels[catid], "probability": float(prob)} for prob, catid in zip(top5_prob, top5_catid)]

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

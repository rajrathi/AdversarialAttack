import torch
import torchvision.models as models
from torchvision import transforms
from functools import lru_cache

MODEL_NAMES = {
    "ResNet18": models.resnet18,
    "EfficientNet_B0": models.efficientnet_b0,
    "MobileNetV2": models.mobilenet_v2
}

@lru_cache(maxsize=3)
def get_model(name):
    model = MODEL_NAMES[name](pretrained=True)
    model.eval()
    return model

def get_imagenet_labels():
    # Download or load ImageNet class labels
    import json, requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(url).text.splitlines()
    return labels

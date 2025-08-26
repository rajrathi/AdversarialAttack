#!/usr/bin/env python3
"""
Model Download Script for Adversarial Attack Demo

This script downloads and caches all the pretrained models used in the demo
to avoid delays during runtime and ensure smooth user experience.
"""

import torch
import torchvision.models as models
import os
import sys
from pathlib import Path

def download_model(model_name, model_func):
    """Download and cache a specific model."""
    print(f"📥 Downloading {model_name}...")
    try:
        # Download with pretrained weights
        model = model_func(pretrained=True)
        model.eval()
        print(f"✅ {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {str(e)}")
        return False

def main():
    """Download all models used in the adversarial attack demo."""
    print("🚀 Starting model download process...")
    print("This may take a few minutes depending on your internet connection.\n")
    
    # Models used in the demo
    models_to_download = {
        "ResNet18": models.resnet18,
        "EfficientNet_B0": models.efficientnet_b0,
        "MobileNetV2": models.mobilenet_v2
    }
    
    success_count = 0
    total_models = len(models_to_download)
    
    # Create cache directory if it doesn't exist
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Models will be cached in: {cache_dir}\n")
    
    # Download each model
    for model_name, model_func in models_to_download.items():
        if download_model(model_name, model_func):
            success_count += 1
        print()  # Add spacing between downloads
    
    # Summary
    print("=" * 50)
    print(f"📊 Download Summary:")
    print(f"✅ Successfully downloaded: {success_count}/{total_models} models")
    
    if success_count == total_models:
        print("🎉 All models downloaded successfully!")
        print("You can now run the adversarial attack demo without delays.")
    else:
        print(f"⚠️  {total_models - success_count} models failed to download.")
        print("The app may experience delays when loading these models.")
    
    print("\n🚀 Ready to run the demo!")
    print("Start the backend: uvicorn backend.main:app --reload")
    print("Start the frontend: streamlit run frontend/app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

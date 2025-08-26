import torch
import random

def patch(input_tensor, patch_size=32):
    patched = input_tensor.clone()
    c, h, w = patched.shape[1:]
    y = random.randint(0, h-patch_size)
    x = random.randint(0, w-patch_size)
    patched[0, :, y:y+patch_size, x:x+patch_size] = 1.0
    return patched

import torch
import random

def sp_noise(input_tensor, noise_level):
    noisy = input_tensor.clone()
    c, h, w = noisy.shape[1:]
    num_pixels = int(noise_level * h * w)
    for _ in range(num_pixels):
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        val = random.choice([0., 1.])
        noisy[0, :, y, x] = val
    return noisy

import torch
import torch.nn.functional as F

def blur(input_tensor, kernel_size):
    # Simple Gaussian blur using conv2d
    channels = input_tensor.shape[1]
    kernel = torch.ones((channels, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
    blurred = F.conv2d(input_tensor, kernel, padding=kernel_size//2, groups=channels)
    return blurred

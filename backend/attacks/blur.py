import torch
import torch.nn.functional as F
import math

def gaussian_kernel_2d(kernel_size, sigma):
    """Generate a 2D Gaussian kernel."""
    # Create coordinate grids
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    
    # Create meshgrid
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    
    # Calculate Gaussian values
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    return kernel

def blur(input_tensor, kernel_size):
    """
    Apply Gaussian blur to input tensor.
    
    Args:
        input_tensor: Input image tensor of shape (batch, channels, height, width)
        kernel_size: Size of the Gaussian kernel (should be odd)
    
    Returns:
        Blurred tensor of the same shape as input
    """
    try:
        # Ensure kernel size is odd and at least 3
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Get tensor properties
        batch_size, channels, height, width = input_tensor.shape
        device = input_tensor.device
        
        # Calculate sigma based on kernel size (common heuristic)
        sigma = kernel_size / 6.0
        
        # Generate Gaussian kernel
        kernel_2d = gaussian_kernel_2d(kernel_size, sigma)
        
        # Expand kernel for convolution
        # Shape: (channels, 1, kernel_size, kernel_size)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, kernel_size, kernel_size)
        kernel = kernel.to(device)
        
        # Apply padding to maintain image size
        padding = kernel_size // 2
        
        # Apply blur using grouped convolution (each channel separately)
        blurred = F.conv2d(
            input_tensor, 
            kernel, 
            padding=padding, 
            groups=channels
        )
        
        # Ensure output is in valid range [0, 1]
        blurred = torch.clamp(blurred, 0, 1)
        
        return blurred
        
    except Exception as e:
        print(f"Error in Gaussian blur: {e}")
        # Fallback: return original tensor if blur fails
        return input_tensor.clone()

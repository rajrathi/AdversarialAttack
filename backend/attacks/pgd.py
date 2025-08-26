import torch

def pgd(model, input_tensor, epsilon, steps):
    """
    Projected Gradient Descent attack
    """
    # Create a copy of the input tensor
    perturbed = input_tensor.clone().detach()
    
    # Store original input for projection
    original = input_tensor.clone().detach()
    
    for step in range(steps):
        # Enable gradient computation
        perturbed.requires_grad_(True)
        
        # Forward pass
        output = model(perturbed)
        
        # Create target (use predicted class as target for untargeted attack)
        target = output.argmax(dim=1)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Zero gradients
        model.zero_grad()
        if perturbed.grad is not None:
            perturbed.grad.zero_()
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        data_grad = perturbed.grad.data
        
        # Apply gradient sign
        perturbed = perturbed.detach() + (epsilon / steps) * data_grad.sign()
        
        # Project back to epsilon ball around original input
        delta = perturbed - original
        delta = torch.clamp(delta, -epsilon, epsilon)
        perturbed = original + delta
        
        # Clip to valid image range
        perturbed = torch.clamp(perturbed, 0, 1)
    
    return perturbed.detach()

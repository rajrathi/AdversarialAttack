import torch

def pgd(model, input_tensor, epsilon, steps):
    perturbed = input_tensor.clone().detach()
    perturbed.requires_grad = True
    for _ in range(steps):
        output = model(perturbed)
        loss = torch.nn.functional.cross_entropy(output, output.argmax(dim=1))
        model.zero_grad()
        loss.backward()
        data_grad = perturbed.grad.data
        perturbed = perturbed + epsilon * data_grad.sign()
        perturbed = torch.clamp(perturbed, 0, 1)
        perturbed = perturbed.detach()
        perturbed.requires_grad = True
    return perturbed

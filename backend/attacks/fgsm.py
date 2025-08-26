import torch

def fgsm(model, input_tensor, epsilon):
    input_tensor.requires_grad = True
    output = model(input_tensor)
    loss = torch.nn.functional.cross_entropy(output, output.argmax(dim=1))
    model.zero_grad()
    loss.backward()
    data_grad = input_tensor.grad.data
    perturbed = input_tensor + epsilon * data_grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed.detach()

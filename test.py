import torch
x = torch.rand(1, 3, 224, 224)
print(torch.nn.functional.relu(x))
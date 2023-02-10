import torch
import matplotlib.pyplot as plt

def to_numpy(x):
    return x.detach().cpu().numpy()

x = torch.linspace(0, 10, 100)
x.requires_grad = True

y = torch.sin(x)

plt.plot(to_numpy(y), label='sin(x)')

y_deriv = torch.autograd.grad(y, x, torch.ones_like(y))[0]
plt.plot(to_numpy(y_deriv), label="sin'(x)"); plt.legend()


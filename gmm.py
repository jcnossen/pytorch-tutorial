# 1D Gaussian mixture model with two components
import torch
import matplotlib.pyplot as plt
import numpy as np
from array_view import array_plot

def to_numpy(x):
    return x.detach().cpu().numpy()

def normal_prob(x, mu, std):
    return torch.exp(-0.5 * ( (x - mu) / std )**2) / (std * np.sqrt(2*torch.pi))

# Define the 2 gaussian distributions
true_weights = torch.tensor([0.3, 0.7])
true_means = torch.tensor([3, 8], dtype=torch.float32)
true_std = torch.tensor([0.5, 2], dtype=torch.float32)
N = 2000 # Number of samples

# Generate samples 
smp_comp = torch.multinomial(true_weights, N, replacement=True)
samples = torch.normal(true_means[smp_comp], true_std[smp_comp])

# Plot our sample data
plt.hist(to_numpy(samples), bins=100, range=[0, 15], density=True)

# Problem parameters to estimate
est_weights = torch.ones(2)*0.5
est_means = torch.tensor([1, 10],dtype=torch.float32)
est_std = torch.tensor([2,2],dtype=torch.float32)
parameters = [est_weights, est_means, est_std]
for p in parameters:
    p.requires_grad = True

# Create an optimizer
optimizer = torch.optim.Adam(parameters, lr=0.1)

result = []
for i in range(100):
    optimizer.zero_grad()
    
    # Calculate the probability of each sample for each normal distribution, size [N, 2]
    smp_prob_per_comp = normal_prob(samples[:,None], est_means[None], est_std[None])
    
    # Probability per sample, size [N]
    smp_prob = ( (est_weights / est_weights.sum())[None] * smp_prob_per_comp).sum(1)
    
    # Define the total log-probability for the GMM, 
    # iaw probability of getting all these samples with current estimated distribution
    total_log_prob = -torch.log(smp_prob).sum() # The optimizer will minimize, so we minimize the negative log-prob
    total_log_prob.backward()
    print(f"Total log-prob: {to_numpy(total_log_prob):.2f}. est_weights:{to_numpy(est_weights)}")

    optimizer.step()
    with torch.no_grad():
        est_weights.clamp_(0,1)
        est_weights /= est_weights.sum()
        
    x = torch.linspace(0, 15, steps=200)
    result.append( to_numpy( est_weights[None] * normal_prob(x[:,None], est_means[None], est_std[None]) ).sum(1) )

        
with torch.no_grad():
    x = torch.linspace(0, 15, steps=200)
    for i in range(2):
        plt.plot(x, to_numpy( est_weights[i] * normal_prob(x, est_means[i], est_std[i]) ) )

array_plot(result)



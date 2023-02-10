# 1D Gaussian mixture model with two components
import torch
import matplotlib.pyplot as plt
import numpy as np
from array_view import array_plot

def to_numpy(x):
    return x.detach().cpu().numpy()

def normal_prob(x, mu, std):
    return torch.exp(-0.5 * ( (x - mu) / std )**2) / (std * np.sqrt(2*torch.pi))

class GMM(torch.nn.Module):
    """
    Calculates probability in a gaussian mixture model
    """
    def __init__(self, weights, means, std):
        super().__init__()

        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.means = torch.nn.Parameter(means, requires_grad=True)
        self.std = torch.nn.Parameter(std, requires_grad=True)
        
    def forward(self, samples):
        # Calculate the probability of each sample for each normal distribution, size [N, 2]
        smp_prob_per_comp = normal_prob(samples[:,None], self.means[None], self.std[None])
        
        # Probability per sample, size [N]
        smp_prob = ( (self.weights / self.weights.sum())[None] * smp_prob_per_comp).sum(1)
        
        return smp_prob
        
    def sample(self, num_samples):
        # Generate samples 
        with torch.no_grad():
            smp_comp = torch.multinomial(self.weights, num_samples, replacement=True)
            samples = torch.normal(self.means[smp_comp], self.std[smp_comp])
        return samples, smp_comp

    def normalize_weights(self):
        with torch.no_grad():
            self.weights.clamp_(0,1)
            self.weights /= self.weights.sum()
        


# Define the 2 gaussian distributions
true_weights = torch.tensor([0.3, 0.7])
true_means = torch.tensor([3, 8], dtype=torch.float32)
true_std = torch.tensor([0.5, 2], dtype=torch.float32)
N = 2000 # Number of samples

true_gmm = GMM(true_weights, true_means, true_std)

samples, smp_comp = true_gmm.sample(N)

# Plot our sample data
plt.hist(to_numpy(samples), bins=100, range=[0, 15], density=True)

est_weights = torch.ones(2)*0.5
est_means = torch.tensor([1, 10],dtype=torch.float32)
est_std = torch.tensor([2,2],dtype=torch.float32)

est_gmm = GMM(est_weights, est_means, est_std)

# Showing that .parameters() now gives the all the GMM parameters that need to be estimated
print(list(est_gmm.parameters()))

# Create an optimizer
optimizer = torch.optim.Adam(est_gmm.parameters(), lr=0.1)

result = []
for i in range(100):
    optimizer.zero_grad()
    
    smp_prob = est_gmm.forward(samples)
    
    # Define the total log-probability for the GMM, 
    # iaw probability of getting all these samples with current estimated distribution
    total_log_prob = -torch.log(smp_prob).sum() # The optimizer will minimize, so we minimize the negative log-prob
    total_log_prob.backward()
    print(f"Total log-prob: {to_numpy(total_log_prob):.2f}. est_weights:{to_numpy(est_weights)}")

    optimizer.step()
    
    est_gmm.normalize_weights()
        
    x = torch.linspace(0, 15, steps=200)
    result.append( to_numpy( est_gmm.forward(x) ) )

        
with torch.no_grad():
    x = torch.linspace(0, 15, steps=200)
    for i in range(2):
        plt.plot(x, to_numpy( est_gmm.forward(x) ) )

array_plot(result)



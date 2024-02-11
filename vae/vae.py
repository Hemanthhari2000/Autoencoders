import torch 
from torch import nn
import torch.nn.functional as F

# Input Image -> Hidden Image -> mean, std, -> Parameterization Trick -> Decoder -> Output Image
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20) -> None:
        super().__init__()

        #Encoder
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.sigma_layer = nn.Linear(hidden_dim, z_dim)

        #Decoder
        self.layer2 = nn.Linear(z_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()


    def encode(self, x): 
        # q_phi(z|x)
        h = self.relu(self.layer1(x))
        mu, sigma = self.mu_layer(h), self.sigma_layer(h)   
        return mu, sigma
    

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.layer2(z))
        return torch.sigmoid(self.layer3(h))

    def forward(self, x): 
        mu, sigma = self.encode(x)
        std = torch.exp(0.5 * sigma)    
        epsilon = torch.randn_like(std)
        z_reparameterized =  epsilon*sigma + mu
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed, mu, sigma
    


if __name__ == "__main__": 
    x = torch.randn(4, 28*28)
    print(f"Input: {x}")

    vae_model = VariationalAutoencoder(input_dim=28*28)  
    x_reconstructed, mu, sigma = vae_model(x)

    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)

    
        
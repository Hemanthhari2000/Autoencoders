import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Preprocessing data
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 36), 
            nn.ReLU(),
            nn.Linear(36, 18), 
            nn.ReLU(),
            nn.Linear(18, 9)
        )

        self.decoder = nn.Sequential(
            nn.Linear(9, 18), 
            nn.ReLU(),
            nn.Linear(18, 36), 
            nn.ReLU(),
            nn.Linear(36, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, 28*28), 
            nn.Sigmoid(),           
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    

autoencoder_model = Autoencoder()

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=1e-1, weight_decay=1e-8)


epochs = 20
outputs = []    
losses = []

for epoch in tqdm(range(epochs)):
    for (image, _) in loader:
        image = image.reshape(-1, 28*28)
        
        reconstructed_image = autoencoder_model(image)

        loss =  loss_function(reconstructed_image, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    outputs.append((epoch, image, reconstructed_image))

curr_timestamp = str(time.time()).split(".")[0]

if not os.path.exists(f"./data/metrics/{curr_timestamp}/"):
    os.makedirs(f"./data/metrics/{curr_timestamp}/")
torch.save(losses, f'./data/metrics/{curr_timestamp}/losses.pt')
torch.save(outputs, f"./data/metrics/{curr_timestamp}/outputs.pt")




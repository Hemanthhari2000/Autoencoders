import os
import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import ceil

from config import *

from vae import VariationalAutoencoder


class VAE:
    def __init__(self) -> None:
        self.dataset = datasets.MNIST(
            root="./dataset", train=True, download=True, transform=transforms.ToTensor()
        )
        self.train_loader = DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        self.model = VariationalAutoencoder(INPUT_DIM, HIDDEN_DIM, Z_DIM).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_RATE)
        self.loss_fn = nn.BCELoss(reduction="sum")
        self.curr_timestamp = ""
        self.curr_timestamp = str(time.time()).split(".")[0]

    def train_with_weights_from(self, model_path="./data/metrics/1707651055/model.pt"):
        self._load_model(model_path)
        self._train()

    def train(self):
        self._train()

    def inference(self, digit, num_of_examples, model_path):
        self._inference(digit, num_of_examples, model_path)

    def _kl_divergence_loss(self, mu, sigma):
        return -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

    # Training
    def _train(self):
        for _ in range(NUM_EPOCS):
            loop = tqdm(enumerate(self.train_loader))
            for i, (x, _) in loop:
                x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
                x_reconstructed, mu, sigma = self.model(x)

                reconstruction_loss = self.loss_fn(x_reconstructed, x)
                kl_div = self._kl_divergence_loss(mu, sigma)

                loss = reconstruction_loss + kl_div
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loop.set_postfix(loss=loss.item())
            self._save_model()

    def _inference(self, digit, num_of_examples, model_path):
        images = []
        reconstructed_images = []
        idx = 0
        for image, label in self.dataset:
            if label == idx:
                images.append(image)
                idx += 1
            if idx == 10:
                break

        self._load_model(model_path)
        encodings_digit = []
        for d in range(10):
            with torch.no_grad():
                mu, sigma = self.model.encode(images[d].to(DEVICE).view(1, 28 * 28))
            encodings_digit.append((mu, sigma))

        mu, sigma = encodings_digit[digit]
        print("Digit is : ", digit)
        for _ in range(num_of_examples):
            epsilon = torch.exp(0.5 * torch.randn_like(sigma))
            z = epsilon * sigma + mu
            output = self.model.decode(z)
            output = output.view(-1, 1, 28, 28)
            reconstructed_images.append(output)
        print("Length of reconstructed images: ", len(reconstructed_images))
        self._plot_image(reconstructed_images, digit)

    def _plot_image(self, images, digit):
        nrow, ncol = ceil(len(images) / 5), 5
        _, axis = plt.subplots(nrow, ncol)
        axis = axis.flatten()
        for img, axs in zip(images, axis):
            axs.imshow(img.detach().cpu().reshape(28, 28), cmap="gray")
        plt.show()

    def _save_model(self):
        if not os.path.exists(f"./data/metrics/{self.curr_timestamp}/"):
            os.makedirs(f"./data/metrics/{self.curr_timestamp}/")
        torch.save(self.model, f"./data/metrics/{self.curr_timestamp}/model.pt")
        print(f"Model Saved at ./data/metrics/{self.curr_timestamp}/model.pt")

    def _load_model(self, model_path):
        self.model = torch.load(model_path).to(DEVICE)
        print("Model Loaded Successfully from ", model_path)


vae_model = VAE()
### Train model without loading the model weights
# vae_model.train()

### Train model by loading the model weights
# vae_model.train_with_weights_from(model_path="./data/metrics/1707656652/model.pt")

### Model Inference
vae_model.inference(
    digit=5, num_of_examples=6, model_path="./data/metrics/1707657219/model.pt"
)

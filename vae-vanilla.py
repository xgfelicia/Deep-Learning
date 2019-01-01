
# WIP

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torchvision


# take MNIST data and tranform input values from [0, 255] to [-1, 1]
def mnist():
    out_dir = './dataset'
    train = datasets.MNIST(root = out_dir, train = True, transform = transforms.ToTensor(), download = True)
    test = datasets.MNIST(root = out_dir, train = False, transform = transforms.ToTensor())
    return train, test


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def generate_image(vae):
    z = torch.randn(64, 2)
    sample = vae.decoder(z)

    torchvision.utils.save_image(sample.view(64, 1, 28, 28), './sample_vae' + '.png')


##########################################################
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode_layer = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, 2)
        self.log_var = nn.Linear(256, 2)

        self.decode_layer = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )


    def encoder(self, x):
        x = self.encode_layer(x)
        return self.mu(x), self.log_var(x)

    def decoder(self, x):
        return self.decode_layer(x)

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 28*28))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z

#####################################################

def training(vae, optimizer, epoch, train_loader):
    vae.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))



if __name__ == '__main__':
    size = 100
    train, test = mnist()
    train_loader = torch.utils.data.DataLoader(train, batch_size = size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = size, shuffle = False)

    vae = VAE()
    optimizer = optim.Adam(vae.parameters())

    # training
    for epoch in range(0, 20):
        training(vae, optimizer, epoch, train_loader)

    generate_image(vae)

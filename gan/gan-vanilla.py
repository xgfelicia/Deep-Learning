
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
    compose = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    out_dir = '../dataset'
    return datasets.MNIST(root = out_dir, train = True, transform = compose, download = True)

def img_to_vec(images):
    return images.view(images.size(0), 28 * 28)

def vec_to_img(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


##############################################################
# takes image input and decides if it is real or fake


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 28 * 28 # 784
        n_out = 1

        self.layer1 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()  # range [0, 1]
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


############################################################
# generates fake images

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 100
        n_out = 28 * 28

        self.layer1 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()               # [-1, 1] range
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


#############################################################
def train_discriminator(discriminator, optimizer, real_data, fake_data):
    n = real_data.size(0)
    loss = nn.BCELoss() # binary cross entropy loss
    optimizer.zero_grad()

    predict_real = discriminator.forward(real_data)
    err_real = loss(predict_real, Variable(torch.ones(n, 1)))
    err_real.backward()

    predict_fake = discriminator.forward(fake_data)
    err_fake = loss(predict_fake, Variable(torch.zeros(n, 1)))
    err_fake.backward()

    optimizer.step()

    return err_real + err_fake, predict_real, predict_fake


def train_generator(discriminator, optimizer, fake_data):
    n = fake_data.size(0)
    loss = nn.BCELoss()
    optimizer.zero_grad()

    predict = discriminator(fake_data)
    error = loss(predict, Variable(torch.ones(n, 1)))
    error.backward()

    optimizer.step()

    return error

#############################################################

if __name__ == '__main__':
    size = 100
    data = mnist()
    loader = torch.utils.data.DataLoader(data, batch_size = size, shuffle = True)
    num_batches = len(loader)

    d_net = Discriminator()
    g_net = Generator()
    d_opt = optim.Adam(d_net.parameters(), lr = 0.0002)
    g_opt = optim.Adam(g_net.parameters(), lr = 0.0002)

    epochs = 100
    for epoch in range(0, epochs):
        for n_batch, (real_batch,_) in enumerate(loader):
            n = real_batch.size(0)

            # train discriminator
            real_data = Variable(img_to_vec(real_batch))

            fake_vec = Variable(torch.randn(n, 100))
            fake_data = g_net.forward(fake_vec).detach()

            d_error, d_pred_real, d_pred_fake = train_discriminator(d_net, d_opt, real_data, fake_data)

            # train generator
            fake_data = g_net.forward(fake_vec)
            g_error = train_generator(d_net, g_opt, fake_data)

            # print error
            print(epoch, n_batch, "Errors:", d_error.data, g_error.data)

            '''
            # display progress
            if n_batch % 100 == 0:
                vectors = Variable(torch.randn(n, 100))
                test_images = vec_to_img(g_net(vectors))
                torchvision.utils.save_image(test_images.view(-1, 1, 28, 28),
                                                './sample-gan-' + str(n_batch) + '.png')
            '''

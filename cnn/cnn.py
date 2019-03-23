# implementing CNN in PyTorch

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import argparse


# get device CUDA or CPU
parser = argparse.ArgumentParser(description = "MNIST Testing")
parser.add_argument('--no-cuda', action = 'store_true', default = False)

ARGS = parser.parse_args()

use_cuda = torch.cuda.is_available() and not ARGS.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)




# import CIFAR10 data after transformation
# transform PIL Images of [0,1] to tensors of normalized range [-1, 1]
def importCIFAR(pin_mem):
    transform = transforms.Compose( [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True,
                                                    pin_memory = pin_mem, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False,
                                                    pin_memory = pin_mem, num_workers = 2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

###################################################################33

# a neural network with configuration comes from pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.run1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.run2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        x = self.run1(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.run2(x)

        return x


    def train(self, dataloader, optimizer, epochs):
        cn_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):

            for i, data in enumerate(dataloader):
                inputs, labels = data

                outputs = self.forward(inputs.to(device))

                loss = cn_loss(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % 500 == 0:
                    print(epoch + 1, i, loss)


    def test(self, dataloader):
        correct = 0
        total = 0

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data

                outputs = self.forward(inputs.to(device))
                # returns max in each dimension 1 and the argmax values
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()


            print(correct / total)



if __name__ == '__main__':
    trainloader, testloader, classes = importCIFAR(use_cuda)
    epochs = 5

    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

    net.train(trainloader, optimizer, epochs)
    net.test(testloader)

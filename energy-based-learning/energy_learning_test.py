# notes: originally erik's code, I only made some modifications 

import torch
from torch import nn

import numpy as np

# Three arguments:
#   weights: (theta) parameters of the convnet
#   U(x; theta): energy, indirectly depends on parameters theta
#   x: sample

# Two gradients:
#   langevin steps update
#   gradient to update theta

class EnergyModel(nn.Module):
    def __init__(self, size):
        super(EnergyModel, self).__init__()
        self.size = size
        self.lr = 0.0005

        # initialize energy model parameters theta
        self.layer1 = nn.Linear(self.size, 1000)
        self.layer2 = nn.Linear(1000, 500)
        self.layer3 = nn.Linear(500, self.size)

        self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)

    # careful to add activation or else it becomes regular linear multiplication
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x


    # energy function
    def U(self, x):
        return -1.0 * self.forward(x).sum()


    def go(self, data1, data2, printing = False):
        self.train()

        # initialize variables from positive sample and negative sample of normal distribution
        # x_pos comes with sample data (X_train)
        # x_neg needs to be sampled from the starting values (X_demo)
        x_pos = torch.autograd.Variable(torch.tensor(data1, dtype = torch.float32), requires_grad = True)
        x_neg = torch.autograd.Variable(torch.tensor(data2, dtype = torch.float32), requires_grad = True)

        if printing:
            print(self.U(x_pos), self.U(x_neg))

        # update samples via stochastic gradient Langevin dynamics
        # use langevin to sample -- use x_neg as starting point and update it with gradient after every step
        # gradient of the energy function
        langevin_step_size = 0.01
        langevin_steps = 10
        for j in range(langevin_steps):
            x_neg_grad = torch.autograd.grad(self.U(x_neg), x_neg)[0] # given theta and data
            # minimize x_neg to maximize its energy U()
            noise = torch.sqrt(torch.tensor(2.0 * langevin_step_size)) * torch.randn_like(x_neg)
            x_neg.data = x_neg.data + (-0.5 * langevin_step_size) * (x_neg_grad.data) + (noise)

        if printing:
            print(self.U(x_pos), self.U(x_neg))

        # goal: minimize energy by minimizing U(x_pos) and maximizing U(x_neg) [max x_pos and min x_neg]
        #       x_pos is the sample data
        #       x_neg is the data from repeated langevin steps
        # apply energy to each dataset to discover an energy function that
        # assigns low values to high probability events and vice versa
        self.optim.zero_grad()
        loss = self.U(x_pos) - self.U(x_neg)
        loss.backward()
        self.optim.step() # updating parameters
        self.eval()

        print("Loss: ", loss)



def main():
    size = 1000
    epoch = 200
    mean1 = 0
    mean2 = 10
    var1 = 10
    var2 = 50

    model = EnergyModel(size)

    for i in range(epoch):
        sample = np.random.normal(mean1, var1, size)
        sample2 = np.random.normal(mean2, var2, size)
        sample = np.sort(sample)
        sample2 = np.sort(sample2)
        model.go(sample, sample2, printing = False)


if __name__ == '__main__':
    main()

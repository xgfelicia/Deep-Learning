# guide: https://skymind.ai/wiki/restricted-boltzmann-machine
#        https://heartbeat.fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class RBM():
    def __init__(self, visible, hidden):
        # two parameters: visible = number of visible nodes
        #                 hidden = number of hidden nodes
        self.W = torch.randn(hidden, visible)
        self.a = torch.randn(1, hidden)      # P(hidden nodes | visible nodes)
        self.b = torch.randn(1, visible)     # P(visible nodes | hidden nodes)


    # purpose: sample hidden nodes from visible nodes x
    #          then compute the P(hidden nodes | visible nodes)
    def sample_hidden(self, x):
        wx = x @ self.W.t()
        wx = wx.view((1, self.W.shape[0]))

        # sigmoid activation function with (weights * x) + bias
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)

        return p_h_given_v, torch.bernoulli(p_h_given_v)


    # purpose: sample visible nodes from hidden nodes y
    #          then compute the P(visible nodes | hidden nodes)
    def sample_visible(self, y):
        wy = y @ self.W
        wy = wy.view((1, self.W.shape[1]))

        # sigmoid activation function with as (weights * y) + bias
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)


    # purpose: update parameters after trainin
    # arguments: v0 = initial input vector
    #            vk = visible nodes after k samplings
    #            ph0 = initial vector of probabilities
    #            phk = probabilities of hidden nodes after k samplings
    def update(self, v0, vk, ph0, phk):
        # weights are adjusted through constructive divergence
        self.W += ((v0.t() @ ph0) - (vk.t() @ phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)



def train(size, epoch_num, iterates):
    nb_epoch = epoch_num

    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.

        # generate dataset for every new epoch
        training_set = np.random.normal(5, 10, size)
        training_set = np.where(training_set >= 0, 1, 0)
        training_set = torch.Tensor(np.sort(training_set)).view((1, size))

        vk = training_set
        v0 = training_set
        ph0,_ = rbm.sample_hidden(v0)

        # transform v0 using #iterates of samplings
        # weights for the visible nodes are used to generate the hidden nodes
        # these weights are used to reconstruct the visible nodes
        for k in range(iterates):
            # repeated forwards and backwards passes
            # forward pass: inputs -> prediction of activations
            # backward pass: activations -> reconstructions (prediction of original input)
            _,hk = rbm.sample_hidden(vk)
            _,vk = rbm.sample_visible(hk)

        phk,_ = rbm.sample_hidden(vk)

        # update parameters after samplings
        rbm.update(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0 - vk))

        s += 1.
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss))



##################################################################

if __name__ == '__main__':
    epoch = 50
    size = 100
    iterates = 10

    visible = size
    hidden = visible * 2
    rbm = RBM(visible, hidden)

    train(size, epoch, iterates)

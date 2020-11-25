import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class Recognizer_mlp_cifar(nn.Module):
    def __init__(self, opt, label_index=None):
        super(Recognizer_mlp_cifar, self).__init__()
        self.opt = opt
        self.out_num = 1 
        self.insize = self.opt.img_size // 3
        self.latent1 = 300
        self.latent2 = 300
        self.mlp1_1 = torch.nn.Parameter(torch.Tensor(self.latent1, self.insize))
        self.mlp1_2 = torch.nn.Parameter(torch.Tensor(self.latent1, self.insize))
        self.mlp1_3 = torch.nn.Parameter(torch.Tensor(self.latent1, self.insize))
        self.mlp2 = torch.nn.Parameter(torch.Tensor(self.latent2, 3*self.latent2))
        self.mlp3 = torch.nn.Parameter(torch.Tensor(self.out_num, self.latent2))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.001
        self.mlp1_1.data.uniform_(-stdv, stdv)
        self.mlp1_2.data.uniform_(-stdv, stdv)
        self.mlp1_3.data.uniform_(-stdv, stdv)
        self.mlp2.data.uniform_(-stdv, stdv)
        self.mlp3.data.uniform_(-stdv, stdv)

    def forward(self, img):
        layer1_1 = F.leaky_relu(F.linear(img[:, 0, :], self.mlp1_1))
        layer1_2 = F.leaky_relu(F.linear(img[:, 1, :], self.mlp1_2))
        layer1_3 = F.leaky_relu(F.linear(img[:, 2, :], self.mlp1_3))
        layer1 = torch.cat((layer1_1, layer1_2, layer1_3), 1)
        layer2 = F.leaky_relu(F.linear(layer1, self.mlp2))
        out = F.linear(layer2, self.mlp3)

        return out
    

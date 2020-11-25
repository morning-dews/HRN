import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class Recognizer_mlp(nn.Module):
    def __init__(self, opt, label_index=None):
        super(Recognizer_mlp, self).__init__()
        self.opt = opt
        latent = 100
        self.weight1 = nn.Parameter(torch.Tensor(latent, opt.img_size))
        self.weight2 = nn.Parameter(torch.Tensor(1, latent))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (2000**0.5)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
    
    def forward(self, img):
        layer1 = F.leaky_relu(F.linear(img, self.weight1))
        out = F.linear(layer1, self.weight2)
        return out

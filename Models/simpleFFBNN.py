import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
import KlLayers
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
torch.manual_seed(43)

class SimpleFFBNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleFFBNN, self).__init__()
        self.fc1 = KlLayers.KlLayers(input_dim, 400)
        self.fc2 = KlLayers.KlLayers(400, 200)
        self.fc3 = KlLayers.KlLayers(200, output_dim)

        self.kl_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def kl_divergence(self):
        kl = 0
        for layer in self.kl_layers:
            kl += layer.kl_divergence()
        return kl
        
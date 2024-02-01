import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU
import torch.nn.functional as F
import KlLayers
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
torch.manual_seed(33)
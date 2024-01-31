# imports
import torch
from torch import nn
from torch.nn import functional as F
from torch import Module


def local_reparam_trick(mu, logvar, cuda = False, sample = True):
    """ Function for local reparameterization trick. This is used to sample from the Gaussian distribution
        using the mean and log variance of the distribution. This is used for sampling dropout masks and weights.
        This is based of the paper "INSERT PAPER NAME HERE" by INSERT AUTHORS HERE (INSERT YEAR HERE)"
    """
    if sample:
        std = logvar.mul(0.5).exp_()
        if cuda:
            epsilon = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            epsilon = torch.FloatTensor(std.size()).normal_()
        
        epsilon = torch.autograd.Variable(epsilon)
        return mu + std * epsilon
    else:
        return mu


Class KlLayers(Module):
    """ Class for KL divergence layers. This is an implementation of Fully Connected Group Normal-Jeffrey's layer.
        This is based of the paper "Efficacy of Bayesian Neural Networks in Active Learning" by Rakish & Jain (2017)
    """

    def __init__(self, in_features, out_features, cuda = False, initial_weights = None, initial_bias = None, clip_variance = None):
        super(KlLayers, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda
        self.clip_variance = clip_variance
        self.dropout_mu = nn.Parameter(torch.Tensor(in_features)) # mean of the Gaussian distribution used for sampling dropout masks
        self.dropout_logvar = nn.Parameter(torch.Tensor(in_features)) # log variance of the Gaussian distribution used for sampling dropout masks
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features)) # mean of the Gaussian distribution used for sampling weights
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features)) # log variance of the Gaussian distribution used for sampling weights
        self.bias_mu = nn.Parameter(torch.Tensor(out_features)) # mean of the Gaussian distribution used for sampling biases
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features)) # log variance of the Gaussian distribution used for sampling biases

        # initialize parameters
        self.reset_parameters(initial_weights, initial_bias)

    def reset_parameters(self, initial_weights, initial_bias):
        """ Function for initializing the parameters of the layer. """
        stdv = 1. / math.sqrt(self.weight_mu.size(1))

        self.dropout_mu.data.normal_(1, 1e-2)

        if initial_weights is None:
            self.weight_mu.data = torch.Tensor(initial_weights)
        else:
            self.weight_mu.data.uniform_(0, stdv)

        if initial_bias is None:
            self.bias_mu.data = torch.Tensor(initial_bias)
        else:
            self.bias_mu.data.fill_(0)

        self.dropout_logvar.data.normal_(-9, 1e-2)  # have to figure out why -9, 1e-2  (maybe -9 = log(1e-4))
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)


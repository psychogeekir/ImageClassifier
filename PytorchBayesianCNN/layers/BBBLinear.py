import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import utils
import metrics
from .misc import ModuleWrapper


class BBBLinear(ModuleWrapper):
    # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/layers/BBBLinear.py
    def __init__(self, in_features, out_features, alpha_shape=(1, 1), max_log_alpha=0.0, bias=True, name='BBBLinear', BNN_record_cfg=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # alpha is the variational dropout, it seems here it can only be in the shape (1, 1)
        # but at most it may be in the same size of W. It can also be in the size (1, in_features) or (1, out_features) due to broadcasting
        # the size of alpha affect the computation efficiency
        self.alpha_shape = alpha_shape
        self.weight = Parameter(torch.Tensor(out_features, in_features))  # optimization
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))  # optimization, rho
        self.max_log_alpha = max_log_alpha  # recommend alpha < 1
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))  # optimization
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # initialization
        self.kl_value = metrics.calculate_kl
        self.name = name
        self.record_cfg = BNN_record_cfg
        if self.record_cfg['record_mean_var']:
            self.mean_var_path = self.record_cfg['mean_var_savepath'] + "/{}.txt".format(self.name)

    def reset_parameters(self):
        # override pytorch initialization
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        # x: batch_size x in_features
        # mean weight: out_features x in_features
        # mean: batch_size x out_features = x * W^T
        mean = F.linear(x, self.weight)
        if self.bias is not None:
            mean = mean + self.bias

        # clamp alpha
        self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_log_alpha)

        # sigma weight: out_features x in_features
        # due to broadcasting rule, log_alpha at most can be in the size (out_features x in_features)
        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        # std: batch_size x out_features
        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        if self.training:
            epsilon = std.data.new(std.size()).normal_()  # epsilon: batch_size x out_features
        else:
            epsilon = 0.0
        # Local reparameterization trick
        out = mean + std * epsilon  # batch_size x out_features

        # self has a default training attribute
        # By default all the modules are initialized to train mode (self.training = True),
        # self.training is controlled by model.eval() and model.train()
        if self.record_cfg['record_mean_var'] and self.record_cfg['record_now'] and self.training and \
                self.name in self.record_cfg['record_layers']:
            utils.save_array_to_file(mean.cpu().detach().numpy(), self.mean_var_path, "mean", self.record_cfg)
            utils.save_array_to_file(std.cpu().detach().numpy(), self.mean_var_path, "std", self.record_cfg)

        return out

    def kl_loss(self):
        # use the same average log_alpha for all W, the output is the summation over all weight
        return self.weight.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()


class BBBLinear_ard(ModuleWrapper):
    # https://github.com/HolyBayes/pytorch_ard/blob/master/torch_ard/torch_ard.py
    def __init__(self, in_features, out_features,  max_log_alpha=8.0, ard_init=-10.0, bias=True, name='BBBLinear',
                 BNN_record_cfg=None):
        super(BBBLinear_ard, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # alpha is the variational dropout, it seems here it can only be in the shape (1, 1)
        # but at most it may be in the same size of W. It can also be in the size (1, in_features) or (1, out_features) due to broadcasting
        # the size of alpha affect the computation efficiency

        self.weight = Parameter(torch.Tensor(out_features, in_features))  # optimization
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))  # optimization
        self.max_log_alpha = max_log_alpha
        self.ard_init = ard_init
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))  # optimization
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # initialization
        self.name = name
        self.record_cfg = BNN_record_cfg
        if self.record_cfg['record_mean_var']:
            self.mean_var_path = self.record_cfg['mean_var_savepath'] + "/{}.txt".format(self.name)

    def reset_parameters(self):
        self.weight.data.normal_(std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(0, 0)
        self.log_sigma2.data = self.ard_init * torch.ones_like(self.log_sigma2)

    def forward(self, x):
        # x: batch_size x in_features
        # mean weight: out_features x in_features
        # mean: batch_size x out_features = x * W^T
        mean = F.linear(x, self.weight)
        if self.bias is not None:
            mean = mean + self.bias

        # clamp alpha
        log_alpha = self.clip(self.log_alpha, to=self.max_log_alpha)

        # sigma weight: out_features x in_features
        # due to broadcasting rule, log_alpha at most can be in the size (out_features x in_features)
        sigma = torch.exp(log_alpha) * self.weight * self.weight

        # std: batch_size x out_features
        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        if self.training:
            epsilon = std.data.new(std.size()).normal_()  # epsilon: batch_size x out_features
        else:
            epsilon = 0.0
        # Local reparameterization trick
        out = mean + std * epsilon  # batch_size x out_features

        # self has a default training attribute
        # By default all the modules are initialized to train mode (self.training = True),
        # self.training is controlled by model.eval() and model.train()
        if self.record_cfg['record_mean_var'] and self.record_cfg['record_now'] and self.training and \
                self.name in self.record_cfg['record_layers']:
            utils.save_array_to_file(mean.cpu().detach().numpy(), self.mean_var_path, "mean", self.record_cfg)
            utils.save_array_to_file(std.cpu().detach().numpy(), self.mean_var_path, "std", self.record_cfg)

        return out

    def kl_loss(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.clip(self.log_alpha)
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

    @staticmethod
    def clip(tensor, to=8.0):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    @property
    def log_alpha(self):
        eps = 1e-8
        return self.log_sigma2 - 2 * torch.log(torch.abs(self.weight) + eps)


if __name__ == "__main__":
    pass
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


class BBBConv2d(ModuleWrapper):
    # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/layers/BBBConv.py
    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape, max_log_alpha=1.0, stride=1,
                 padding=0, dilation=1, bias=True, name='BBBConv2d', BNN_record_cfg=None):
        super(BBBConv2d, self).__init__()
        self.kl_value = metrics.calculate_kl
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # alpha is the variational dropout, it seems here it can only be in the shape (1, 1)
        # the size of alpha affect the computation efficiency
        self.alpha_shape = alpha_shape
        self.groups = 1
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.out_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride, self.padding, self.dilation, self.groups)
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.max_log_alpha = max_log_alpha
        self.reset_parameters()
        self.name = name
        self.record_cfg = BNN_record_cfg
        if self.record_cfg['record_mean_var']:
            self.mean_var_path = self.record_cfg['mean_var_savepath'] + "/{}.txt".format(self.name)

    def reset_parameters(self):
        # override pytorch initialization
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x):
        # x: batch_size x in_channels x height x width
        # mean weight: out_channels x in_channels x height x width
        # mean: batch_size x out_channels x height x width
        mean = self.out_bias(x, self.weight)

        # clamp alpha
        self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_log_alpha)

        # sigma weight: out_channels x in_channels x height x width
        # due to broadcasting rule, log_alpha at most can be in the size (out_channels x in_channels x height x width)
        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        # std: batch_size x out_channels x height x width
        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
        if self.training:
            epsilon = std.data.new(std.size()).normal_()  # epsilon: batch_size x out_channels x height x width
        else:
            epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon

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


class BBBConv2d_ard(ModuleWrapper):
    # https://github.com/HolyBayes/pytorch_ard/blob/master/torch_ard/torch_ard.py
    def __init__(self, in_channels, out_channels, kernel_size, max_log_alpha=8.0, ard_init=-10.0, stride=1,
                 padding=0, dilation=1, bias=True, name='BBBConv2d', BNN_record_cfg=None):
        super(BBBConv2d_ard, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.out_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride, self.padding,
                                                       self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride, self.padding, self.dilation,
                                                         self.groups)
        self.log_sigma2 = Parameter(ard_init * torch.ones_like(self.weight))
        self.max_log_alpha = max_log_alpha
        self.ard_init = ard_init
        # self.reset_parameters()
        self.name = name
        self.record_cfg = BNN_record_cfg
        if self.record_cfg['record_mean_var']:
            self.mean_var_path = self.record_cfg['mean_var_savepath'] + "/{}.txt".format(self.name)

    # def reset_parameters(self):
    #     self.weight.data.normal_(std=0.01)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(0, 0)
    #     self.log_sigma2.data = self.ard_init*torch.ones_like(self.log_sigma2)

    def forward(self, x):
        # x: batch_size x in_channels x height x width
        # mean weight: out_channels x in_channels x height x width
        # mean: batch_size x out_channels x height x width
        mean = self.out_bias(x, self.weight)

        # clamp alpha
        log_alpha = self.clip(self.log_alpha, to=self.max_log_alpha)

        # sigma weight: out_channels x in_channels x height x width
        # due to broadcasting rule, log_alpha at most can be in the size (out_channels x in_channels x height x width)
        sigma = torch.exp(log_alpha) * self.weight * self.weight

        # std: batch_size x out_channels x height x width
        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
        if self.training:
            epsilon = std.data.new(std.size()).normal_()  # epsilon: batch_size x out_channels x height x width
        else:
            epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon

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




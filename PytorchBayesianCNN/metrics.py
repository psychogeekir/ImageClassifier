import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class negative_ELBO(nn.Module):
    # minimize is the negative of ELBO, so this is actually minus ELBO
    def __init__(self, train_size, num_batches, reduction='mean', kl_weight=1.0):
        super(negative_ELBO, self).__init__()
        self.train_size = train_size
        self.num_batches = num_batches
        self.reduction = reduction
        self.kl_weight = kl_weight

    def forward(self, input, target, kl):
        assert not target.requires_grad
        # https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html
        # return F.nll_loss(input, target, reduction=self.reduction) * (number of batches) + kl_weight * kl
        # negative log-likelihood + weight * KL divergence
        # in classification problem, input is the logsoftmax of direct output of NN

        negative_loglikelihood = F.nll_loss(input, target, reduction=self.reduction)
        total_loss = negative_loglikelihood + self.kl_weight * kl

        # negative_loglikelihood = F.nll_loss(input, target, reduction=self.reduction) * input.size(0)
        # total_loss = negative_loglikelihood + self.kl_weight * kl / self.num_batches

        # negative_loglikelihood = F.nll_loss(input, target, reduction=self.reduction)
        # total_loss = negative_loglikelihood * self.train_size + self.kl_weight * kl
        return total_loss, negative_loglikelihood


def lr_linear(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(log_alpha):
    # KL divergence: for Gaussian variational posterior and log-uniform prior, KL divergence is not analytically tractable,
    # but ca be approximated
    # Kingma, D. P., Salimans, T., and Welling, M. (2015). “Variational dropout and the local reparameterization trick.”
    #       Adv. Neural Inf. Process. Syst., 2015-January(Mcmc), 2575–2583.
    # - D_KL = constant + 0.5 * log(alpha) + c1 * alpha + c2 * alpha**2 + c3 * alpha**3
    # D_KL
    #
    # D_KL <= constant + 0.5 * log(1/alpha)
    # softplus is more stable when 1/alpha is small
    # D_KL <= constant + 0.5 * log(1 + 1/alpha) = constant + 0.5 * log(1 + exp(-log(alpha)))

    # c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921
    # kl = torch.sum(-(0.5 * log_alpha + c1 * torch.exp(log_alpha) + c2 * torch.exp(log_alpha)**2 + c3 * torch.exp(log_alpha)**3))


    # Molchanov, D., Ashukha, A., and Vetrov, D. (2017). “Variational dropout sparsifies deep neural networks.”
    #       34th Int. Conf. Mach. Learn. ICML 2017, 5, 3854–3863.
    # k1, k2, k3 = 0.63576, 1.8732, 1.48695
    # C = -k1
    # kl = -torch.sum(k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C)

    kl = 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))
    return kl


"""unofficial implementation of SNO
may not correspond to the same exact architecture but the idea is there
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from complexPyTorch.complexLayers import ComplexLinear

from reno.utilities import interp_fourier_2d


def complex_softplus(input):
    return F.softplus(input.real).type(torch.complex64)\
        + 1j * F.softplus(input.imag).type(torch.complex64)


class ComplexMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(ComplexMLP, self).__init__()
        self.lin1 = ComplexLinear(in_channels, mid_channels)
        self.lin2 = ComplexLinear(mid_channels, mid_channels)
        self.lin3 = ComplexLinear(mid_channels, mid_channels)
        self.lin4 = ComplexLinear(mid_channels, out_channels)

    def forward(self, x):
        x = complex_softplus(self.lin1(x))
        x = complex_softplus(self.lin2(x))
        x = complex_softplus(self.lin3(x))
        x = self.lin4(x)
        return x


class TrigonometricSNO(nn.Module):
    def __init__(self, in_channels, out_channels, res_train, pointwise_evals=True, permute=True):
        super(TrigonometricSNO, self).__init__()
        self.permute = permute
        self.pointwise_evals = pointwise_evals
        nhidden = 151
        self.mlp = ComplexMLP(in_channels * res_train[0] * (res_train[1] // 2 + 1),
                              out_channels * res_train[0] * (res_train[1] // 2 + 1), nhidden)
        self.res_train = res_train

    def forward(self, x):
        if self.permute:
            x = x.permute(0, 3, 1, 2)

        orig_res = x.shape[-2:]
        x = interp_fourier_2d(x, self.res_train, permute=False)

        # convert pointwise evals to fourier coefficients
        if self.pointwise_evals:
            c = torch.fft.rfft2(x)
        else:
            c = x

        c = self.mlp(c.reshape(*c.shape[:-2], -1)).reshape(*c.shape)

        if self.pointwise_evals:
            x = torch.fft.irfft2(c)
        else:
            x = c

        x = interp_fourier_2d(x, orig_res, permute=False)

        if self.permute:
            x = x.permute(0, 2, 3, 1)
        return x
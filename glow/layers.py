from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randn
from scipy.linalg import lu, qr


class Bijector(nn.Module):
    """ Bijector Module Abstract Base Class """
    def __init__(self):
        super().__init__()
        self.direction('forward')

    def _forward_fn(self, x, accum=None):
        """ Bijector Forward Function """
        raise NotImplementedError

    def _inverse_fn(self, y, accum=None):
        """ Bijector Inverse Function """
        raise NotImplementedError

    def direction(self, mode='forward'):
        if mode not in ('forward', 'inverse'):
            raise ValueError('bijector direction does not support {}. must be either forward or inverse'.format(mode))
        self._fn = self._inverse_fn if mode == 'inverse' else self._forward_fn
        # TODO mimic nn.Module.train/eval calls children's direction if it exists

    def forward(self, data, accum=None):
        return self._fn(data, accum)


class Block(Bijector):
    """ Block """
    def __init__(self, in_channels, flow_depth=5, num_affine_channels=512, use_lu_optim=False, split_input=True):
        super().__init__()

        squeeze_channels = in_channels * 4

        self.k = flow_depth

        modules = list()
        for i in range(1, self.k+1):
            modules.append(Flow(in_channels, num_affine_channels, use_lu_optim))
        self.flows = nn.ModuleList(modules)

    def _forward_fn(self, data, accum=None):
        for flow in self.flows:
            data, accum = flow(data, accum)
        return data, accum

    def _inverse_fn(self, y, accum=None):
        for flow in reversed(self.flows):
            data, accum = flow(y, accum)
        return data, accum


class Split(nn.Module):
    """ Split """
    def __init__(self):
        super().__init__()

    def forward(self, x, accum=None):
        return x.chunk(2, dim=1), accum


class Squeeze(nn.Module):
    """ Squeeze """
    def __init__(self):
        super().__init__()

    def forward(self, x, y=None, accum=None):
        if y:
            x = torch.concat([x, y], dim=1)
        return x, accum


class Flow(Bijector):
    """  """
    def __init__(self, in_channels, num_affine_channels=512, use_lu_optim=False):
        super().__init__()

        Inv1x1Conv2dBijector = Inv1x1Conv2dLUBijector if use_lu_optim else Inv1x1Conv2dPlainBijector
        convblock = AffineCouplingConv2d(in_channels, num_affine_channels)

        self.flow = nn.Sequential(OrderedDict([
            ('actnorm', ActivationNormalization(in_channels)),
            ('inv1x1', Inv1x1Conv2dBijector(in_channels)),
            ('coupling', AffineCouplingBijector(layer=convblock)),
        ]))

    def _forward_fn(self, x, accum=None):
        """ Bijector Forward Function """
        return self.flow(x, accum)

    def _inverse_fn(self, y, accum=None):
        """ Bijector Inverse Function """
        return self._forward_fn(y, accum)


class ActivationNormalization(Bijector):
    """ ActNorm Layer.

        Activation normalizaton module...
    """
    def __init__(self, channels):
        super().__init__()
        self.initialized = False
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _data_initialization(self, x):
        assert len(x.size()) == 4, 'ActNorm initialization requires inputs of the form (batch, channel, height, width)'

        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], keepdim=True)

            self.shift.data.copy_(mean)
            self.scale.data.copy_(var)

        self.initialized = True

    def _forward_fn(self, x, accum=None):
        """ Forward Function """
        logdet = self._logdet(x)
        accum = accum + logdet if accum else logdet
        x_norm = ((x * self.scale) + self.shift)
        return x_norm, accum

    def _inverse_fn(self, y, accum=None):
        """ Inverse Function """
        logdet = self._logdet(y)
        accum = accum - logdet if accum else logdet
        y_norm = ((y - self.shift) / self.scale)
        return y_norm, accum

    def _logdet(self, x):
        assert len(x.size()) == 4, 'ActNorm module requires inputs of the form (batch, channel, height, width)'
        h, w = x.size(2), x.size(3)
        return h * w * torch.abs(self.scale).log().sum()

    def forward(self, data, accum=None):
        if not self.initialized:
            self._data_initialization(data)
        return self._fn(data, accum)


class Inv1x1Conv2dPlainBijector(Bijector):
    """ Inv1x1Conv2dPlainBijector """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        weight = self._calc_initial_weight()
        self.weight = nn.Parameter(weight)

    def _calc_initial_weight(self):
        rotation = torch.from_numpy(qr(randn(self.channels, self.channels))[0].astype(np.float32))  # sample random rotation matrix
        return rotation

    def _forward_fn(self, x, accum=None):
        y = F.conv2d(x, self.weight)
        logdet = self._logdet(x)
        accum = accum + logdet if accum else logdet
        return y, accum

    def _inverse_fn(self, y, accum=None):
        inverse_weight = torch.inverse(self.weight)
        x = F.conv2d(y, inverse_weight)
        logdet = self._logdet(y)
        accum = accum - logdet if accum else logdet
        return x, accum

    def _logdet(self, x):
        assert len(x.size()) == 4, 'Invertible1by1Conv2d module requires inputs of the form (batch, channel, height, width)'
        h, w = x.size(2), x.size(3)
        return h * w * torch.slogdet(self.weight)[1]


class Inv1x1Conv2dLUBijector(Bijector):
    """ Inv1x1Conv2dLUBijector """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        weight = self._calc_initial_weight()  # initial permutation weight
        # learnable parameters
        self.weight_lower = nn.Parameter(weight['l'])
        self.weight_upper = nn.Parameter(weight['u'])
        self.weight_s = nn.Parameter(weight['s'])
        # persistented parameters
        self.register_buffer('weight_permutation', weight['p'])
        self.register_buffer('s_sign', weight['s_sign'])
        self.register_buffer('fixed_s_sign', weight['s_abslog'])

    def _calc_initial_weight(self):
        rotation = qr(randn(self.channels, self.channels))[0].astype('float32')  # sample random rotation matrix
        permutation, lower, upper = [torch.from_numpy(m) for m in lu(rotation)]
        s = torch.diag(upper)
        s_sign = torch.sign(s)
        s_abs_log = torch.abs(s).log()
        upper = torch.triu(upper, diagonal=1)
        return {
            'p': permutation,
            'u': upper,
            'l': lower,
            's': s,
            's_sign': s_sign,
            's_abslog': s_abs_log,
        }

    def _calc_weight(self):
        return torch.matmul(
            torch.matmul(self.weight_permutation, self.weight_lower),
            torch.add(self.weight_upper + self.weight_s)
        )

    def _forward_fn(self, x, accum=None):
        weight = self._calc_weight()
        y = F.conv2d(x, weight)
        logdet = self._logdet()
        accum = accum + logdet if accum else logdet
        return y, accum

    def _inverse_fn(self, y, accum=None):
        weight = self._calc_weights()
        inverse_weight = torch.inverse(weight)
        x = F.conv2d(y, inverse_weight)
        logdet = self._logdet(y)
        accum = accum - logdet if accum else logdet
        return x, accum

    def _logdet(self):
        return torch.abs(self.weight_s).log().sum()


class ZeroInitConv2d(nn.Module):
    """ ZeroInitConv2d. """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, scale=3.0):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2d.weight.data.zero_()
        self.conv2d.bias.data.zero_()

        self.scale = scale
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def reset_parameters(self):
        self.conv2d.weight.data.zero_()
        self.conv2d.bias.data.zero_()
        self.logs.zero_()

    def forward(self, x):
        x = self.conv2d(x)
        x = x * torch.exp(self.logs * self.scale)
        return x


class AffineCouplingConv2d(nn.Module):
    """ AffineCouplingConv2d """
    def __init__(self, in_channels, num_channels=512, scale=3.0):
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels

        self.block = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels // 2, self.num_channels, 3, padding=1)),
            ('relu_1', nn.ReLU(True)),
            ('conv_2', nn.Conv2d(self.num_channels, self.num_channels, 1, padding=0)),
            ('relu_2', nn.ReLU(True)),
            ('conv_3', ZeroInitConv2d(self.num_channels, self.num_channels, 3, padding=0, scale=scale))
        ]))

        self.block['conv_1'].weight.data.normal_(0, 0.05)
        self.block['conv_1'].bias.data.zero_()

        self.block['conv_3'].weight.data.normal_(0, 0.05)
        self.block['conv_3'].bias.data.zero_()

    def forward(self, x):
        return self.layer(x)


class AffineCouplingBijector(Bijector):
    """ AffineCouplingBijector """
    def __init__(self, layer):
        super.__init__()
        self.layer = layer

    def _forward_fn(self, x, accum=None):
        assert len(x.size()) == 4, 'AffineCouplingBijector module requires inputs of the form (batch, channel, height, width)'
        # forward fn
        num_channels = x.size(1)
        channel_split_size = num_channels // 2
        x_a, x_b = torch.split(x, channel_split_size)
        s_log, t = self.layer(x_b)
        s = torch.exp(s_log)
        y_a = torch.mul(s, x_a) + t
        y_b = x_b
        y = torch.concat(y_a, y_b, dim=1)
        # log determinant
        logdet = self._logdet(s)
        accum = accum + logdet if accum else logdet
        return y, accum

    def _inverse_fn(self, y, accum=None):
        assert len(y.size()) == 4, 'AffineCouplingBijector module requires inputs of the form (batch, channel, height, width)'
        # inverse fn
        num_channels = y.size(1)
        channel_split_size = num_channels // 2
        y_a, y_b = torch.split(y, channel_split_size)
        s_log, t = self.layer(y_b)
        s = torch.exp(s_log)
        x_a = (y_a - t) / s
        x_b = y_b
        x = torch.concat(x_a, x_b, dim=1)
        # log determinant
        logdet = self._logdet(y)
        accum = accum - logdet if accum else logdet
        return x, accum

    def _logdet(self, s):
        return torch.abs(s).log().sum()

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

    def _logdet(self):
        """ Bijector Log Determinant of the Absolute Value of the Forward Function Jacobian """
        raise NotImplementedError

    def direction(self, mode='forward'):
        if mode not in ('forward', 'inverse'):
            raise ValueError('bijector direction does not support {}. must be either forward or inverse'.format(mode))
        self._bijector_fn = self._inverse_fn if mode == 'inverse' else self._forward_fn
        # TODO mimic nn.Module.train/eval calls children's direction if it exists

    def forward(self, data, accum=None):
        return self._bijector_fn(data, accum)


class ActivationNormalization(Bijector):
    """ ActNorm Layer.

        Activation normalizaton module...
    """
    def __init__(self, channels):
        super().__init__()
        self.direction('forward')
        self.initialized = False
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _data_initialization(self, x):
        assert len(x.size()) == 4, 'ActNorm initialization requires inputs of the form (batch, channel, height, width)'

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

    def direction(self, mode='forward'):
        if mode not in ('forward', 'inverse'):
            raise ValueError('direction does not support {}. must be either forward or inverse'.format(mode))
        self._normalization_fn = self._inverse_fn if mode == 'inverse' else self._forward_fn
        # TODO mimic nn.Module.train/eval calls children's direction if it exists

    def forward(self, x):
        if not self.initialized:
            self._data_initialization(x)
        return self._normalization_fn(x)


class Inv1x1Conv2dPlainBijector(Bijector):
    """ Inv1x1Conv2dPlainBijector """
    def __init__(self, channels):
        Bijector.__init__()
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

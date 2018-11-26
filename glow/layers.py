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
    def __init__(self, in_channels, num_affine_channels=512, scale=3.0, force_additive=False, use_lu_optim=False):
        super().__init__()

        Inv1x1Conv2dBijector = Inv1x1Conv2dLUBijector if use_lu_optim else Inv1x1Conv2dPlainBijector

        self.flow = nn.Sequential(OrderedDict([
            ('actnorm', ActivationNormalization(in_channels)),
            ('inv1x1', Inv1x1Conv2dBijector(in_channels)),
            ('coupling', AffineCouplingBijector(in_channels, num_affine_channels, scale, force_additive)),
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
    def __init__(self, channels, scale=1.0, eps=1e-6):
        super().__init__()
        self.initialized = False
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._scale = float(scale)
        self._eps = float(1e-6)

    def _data_initialization(self, x):
        assert len(x.size()) == 4, 'ActNorm initialization requires inputs of the form (batch, channel, height, width)'

        with torch.no_grad():
            mean = -1.0 * x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True)
            var = (x + mean).var(dim=3, keepdim=True).var(dim=2, keepdim=True).var(dim=0, keepdim=True)
            logstd = torch.log(self._scale / (torch.sqrt(var) + self._eps))
            self.shift.data.copy_(mean)
            self.scale.data.copy_(logstd)

        self.initialized = True

    def _forward_fn(self, x, accum=None):
        """ Forward Function """
        x_norm = ((x + self.shift) * torch.exp(self.scale))
        logdet = self._logdet(x)
        accum = accum + logdet if isinstance(accum, torch.Tensor) else logdet
        return x_norm, accum

    def _inverse_fn(self, y, accum=None):
        """ Inverse Function """
        y_norm = ((y * torch.exp(-self.scale)) - self.shift)
        logdet = self._logdet(y)
        accum = accum - logdet if isinstance(accum, torch.Tensor) else -logdet
        return y_norm, accum

    def _logdet(self, x):
        assert len(x.size()) == 4, 'ActNorm module requires inputs of the form (batch, channel, height, width)'
        h, w = x.size(2), x.size(3)
        return h * w * self.scale.sum()

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
        y = F.conv2d(x, self.weight.view(self.channels, self.channels, 1, 1))
        logdet = self._logdet(x)
        accum = accum + logdet if isinstance(accum, torch.Tensor) else logdet
        return y, accum

    def _inverse_fn(self, y, accum=None):
        inverse_weight = torch.inverse(self.weight)
        x = F.conv2d(y, inverse_weight.view(self.channels, self.channels, 1, 1))
        logdet = self._logdet(y)
        accum = accum - logdet if isinstance(accum, torch.Tensor) else -logdet
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
        self.lower = nn.Parameter(weight['lower'])
        self.upper = nn.Parameter(weight['upper'])
        self.logdiag = nn.Parameter(weight['logdiag'])
        # persistented parameters
        self.register_buffer('permutation', weight['permutation'])
        self.register_buffer('signdiag', weight['signdiag'])
        self.register_buffer('maskupper', weight['maskupper'])
        self.register_buffer('masklower', weight['masklower'])
        self.register_buffer('eyelower', weight['eyelower'])

    def _calc_initial_weight(self):
        rotation = qr(randn(self.channels, self.channels))[0].astype('float32')  # sample random rotation matrix
        permutation, lower, upper = [torch.from_numpy(m) for m in lu(rotation)]
        diagonal = torch.diag(upper)
        signdiag = torch.sign(diagonal)
        logdiag = torch.abs(diagonal).log()
        upper = torch.triu(upper, diagonal=1)
        maskupper = torch.triu(torch.ones_like(upper), diagonal=1)
        masklower = maskupper.transpose(0, 1)
        eyelower = torch.eye(masklower.size(0))
        return {
            'permutation': permutation,
            'upper': upper,
            'maskupper': maskupper,
            'lower': lower,
            'masklower': masklower,
            'eyelower': eyelower,
            'signdiag': signdiag,
            'logdiag': logdiag
        }

    def _get_weights(self):
        lower = (self.lower * self.masklower) + self.eyelower
        upper = (self.upper * self.maskupper) + torch.diag(self.signdiag * torch.exp(self.logdiag))
        return self.permutation, lower, upper

    def _forward_fn(self, x, accum=None):
        permutation, lower, upper = self._get_weights()
        weight = torch.matmul(permutation, torch.matmul(lower, upper))
        y = F.conv2d(x, weight.view(self.channels, self.channels, 1, 1))
        logdet = self._logdet(x)
        accum = accum + logdet if isinstance(accum, torch.Tensor) else logdet
        return y, accum

    def _inverse_fn(self, y, accum=None):
        permutation, lower, upper = self._get_weights()
        inverse_weight = torch.matmul(torch.inverse(upper), torch.matmul(torch.inverse(lower), torch.inverse(permutation)))
        x = F.conv2d(y, inverse_weight.view(self.channels, self.channels, 1, 1))
        logdet = self._logdet(y)
        accum = accum - logdet if isinstance(accum, torch.Tensor) else -logdet
        return x, accum

    def _logdet(self, x):
        assert len(x.size()) == 4, 'Invertible1by1Conv2d module requires inputs of the form (batch, channel, height, width)'
        h, w = x.size(2), x.size(3)
        return h * w * self.logdiag.sum()


class ZeroInit3x3Conv2d(nn.Module):
    """ ZeroInitConv2d. """
    def __init__(self, in_channels, out_channels, scale=3.0):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2d.weight.data.zero_()
        self.conv2d.bias.data.zero_()

        self.factor = scale
        self.logscale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def reset_parameters(self):
        self.conv2d.weight.data.zero_()
        self.conv2d.bias.data.zero_()
        self.logscale.zero_()

    def forward(self, x):
        x = self.conv2d(x)
        x = x * torch.exp(self.logscale * self.factor)
        return x


class AffineCouplingConv2d(nn.Module):
    """ AffineCouplingConv2d """
    def __init__(self, in_channels, num_channels=512, scale=3.0, force_additive=False):
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.scale = scale
        self.affine = (not force_additive)

        self.block = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels // 2, self.num_channels, 3, padding=1)),
            ('relu_1', nn.ReLU(True)),
            ('conv_2', nn.Conv2d(self.num_channels, self.num_channels, 1, padding=0)),
            ('relu_2', nn.ReLU(True)),
            ('conv_3', ZeroInit3x3Conv2d(self.num_channels, self.num_channels if self.affine else self.num_channels // 2, scale=scale))
        ]))

        self.block.conv_1.weight.data.normal_(0, 0.05)
        self.block.conv_1.bias.data.zero_()

        self.block.conv_2.weight.data.normal_(0, 0.05)
        self.block.conv_2.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class AffineCouplingBijector(Bijector):
    """ AffineCouplingBijector """
    def __init__(self, in_channels, num_channels=512, scale=3.0, force_additive=False):
        super().__init__()
        self.block = AffineCouplingConv2d(in_channels, num_channels, scale, force_additive)
        self.affine = self.block.affine

    def _forward_fn(self, x, accum=None):
        assert len(x.size()) == 4, 'AffineCouplingBijector module requires inputs of the form (batch, channel, height, width)'
        # forward fn
        x_a, x_b = torch.chunk(x, 2, dim=1)  # split along channel axis
        print(f'x_a size {x_a.size()}')
        print(f'x_b size {x_b.size()}')
        if self.affine:
            logdiag, t = self.block(x_b).chunk(2, dim=1)
            print(f'logdiag size {logdiag.size()}')
            print(f't size {t.size()}')
            diag = torch.exp(logdiag)
            y_a = torch.mul(diag, x_a) + t
            # log determinant
            logdet = self._logdet(diag)
        else:
            out = self.block(x_b)
            y_a = x_a + out
            logdet = 0.0
        y_b = x_b
        y = torch.concat(y_a, y_b, dim=1)
        accum = accum + logdet if isinstance(accum, torch.Tensor) else logdet
        return y, accum

    def _inverse_fn(self, y, accum=None):
        assert len(y.size()) == 4, 'AffineCouplingBijector module requires inputs of the form (batch, channel, height, width)'
        # inverse fn
        y_a, y_b = torch.chunk(y, 2, dim=1)
        if self.affine:
            logdiag, t = self.block(y_b)
            diag = torch.exp(logdiag)
            x_a = (y_a - t) / diag
            logdet = self._logdet(y)
        else:
            out = self.block(y_b)
            x_a = y_a - out
            logdet = 0.0
        x_b = y_b
        x = torch.concat(x_a, x_b, dim=1)
        accum = accum - logdet if isinstance(accum, torch.Tensor) else -logdet
        return x, accum

    def _logdet(self, s):
        return torch.abs(s).log().sum()

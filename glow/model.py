import torch.nn as nn

from .layers import Bijector, Block


class Glow(Bijector):
    def __init__(self, in_channels, flow_depth, layer_depth, use_affine_coupling=True, use_lu_optimization=False):
        super().__init__()
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()
        num_channels = in_channels
        for i in range(1, layer_depth):
            self.blocks.append(
                Block(num_channels, flow_depth, use_affine_coupling, use_lu_optimization, input_split=True)
            )
            num_channels *= 2
        self.blocks.append(
            Block(num_channels, flow_depth, use_affine_coupling, use_lu_optimization, input_split=False)
        )

    def _forward_fn(self, x, log_det=None):
        log_proba_sum = 0
        for block in self.blocks:
            x, log_det, log_proba = block(x, log_det)
            if log_proba:
                log_proba_sum += log_proba
        return log_proba_sum, log_det

    def _inverse_fn(self, z):
        raise NotImplementedError

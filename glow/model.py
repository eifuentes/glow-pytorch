import torch.nn as nn

from .layers import Bijector, FlowBlock


class Glow(Bijector):
    def __init__(self, in_channels, num_levels, num_flow=5, actnorm_scale=1.0,
                 num_coupling_channels=512, coupling_scale=3.0,
                 additive_coupling=False, coupling_lu_optim=False,
                 prior_scale=3.0):
        super().__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels

        self.levels = nn.ModuleList()
        num_channels = self.in_channels
        split_input = True
        for i in range(1, self.num_levels):

            self.levels.append(
                FlowBlock(in_channels, num_flow, split_input,
                          actnorm_scale, num_coupling_channels, coupling_scale,
                          additive_coupling, coupling_lu_optim,
                          prior_scale)
            )
            num_channels *= 2

        split_input = False
        self.levels.append(
            FlowBlock(in_channels, num_flow, split_input,
                      actnorm_scale, num_coupling_channels, coupling_scale,
                      additive_coupling, coupling_lu_optim,
                      prior_scale)
        )

    def _forward_fn(self, x, logdet=None):
        log_proba_sum = 0
        for lvl in self.levels:
            x, logdet, log_proba = lvl(x, logdet)
            if log_proba:
                log_proba_sum += log_proba
        return log_proba_sum, logdet

    def _inverse_fn(self, z):
        # TODO implment reverse
        for lvl in reversed(self.levels):
            lvl()

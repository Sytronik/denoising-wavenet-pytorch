import torch
import torch.nn as nn
from .modules import get_conv1d3x1, ResidualConv1dGLU, get_conv1d1x1


class DWaveNet(nn.Module):
    def __init__(self, in_channels,
                 num_layers=30, num_stacks=3,
                 residual_channels=128, gate_channels=128, skip_out_channels=128,
                 last_channels=(2048, 256),
                 kernel_size=3, bias=False, weight_norm=False,
                 ):
        super().__init__()
        assert num_layers % num_stacks == 0
        num_layers_per_stack = num_layers // num_stacks
        self.l_diff = num_stacks * (2**num_layers_per_stack - 1)

        self.first_conv = get_conv1d3x1(in_channels, residual_channels,
                                        bias=bias, weight_normalization=weight_norm)

        self.conv_layers = nn.ModuleList()
        for n_layer in range(num_layers):
            dilation = 2**(n_layer % num_layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels, gate_channels,
                skip_out_channels=skip_out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                causal=False,
                dropout=1 - 0.95,
                weight_normalization=weight_norm,
            )
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.Sequential(
            nn.ReLU(True),
            get_conv1d3x1(skip_out_channels, last_channels[0],
                          bias=bias, weight_normalization=weight_norm),
            nn.ReLU(True),
            get_conv1d3x1(last_channels[0], last_channels[1],
                          bias=bias, weight_normalization=weight_norm),
            get_conv1d1x1(last_channels[1], 1,
                          bias=True, weight_normalization=weight_norm)
        )

    def forward(self, x):
        x = self.first_conv(x)
        skips = None
        for conv in self.conv_layers:
            x, h = conv(x)
            if skips is None:
                skips = h[..., self.l_diff:-self.l_diff]
            else:
                skips += h[..., self.l_diff:-self.l_diff]

        x = skips
        x = self.last_conv_layers(x)

        return x

import torch
import torch.nn as nn
from .sign_power_layer import SignPow


class ConvWithSignPow(nn.Module):
    def __init__(self, conv: nn.Conv2d, init_alpha=0.0, eps=1e-6):
        super().__init__()

        if not isinstance(conv, nn.Conv2d):
            raise TypeError("conv must be an instance of nn.Conv2d")

        self.signpow = SignPow(
            num_channels=conv.in_channels,
            init_alpha=init_alpha,
            eps=eps
        )

        # 기존 conv 그대로 사용
        self.conv = conv

    def forward(self, x):
        x = self.signpow(x)
        x= self.conv(x)
        
        return x
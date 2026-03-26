import torch
import torch.nn as nn
import torch.nn.functional as F

def scaling_func(x, x_prime, eps=1e-6):
    """
    x, x_prime: shape = (B, C, H, W)

    각 (B, C)별로 H, W 축에 대해서만 mean/std를 구해서
    x_prime을 x와 같은 channel-wise 통계로 affine scaling
    x, x_prime 을 연산 그래프에 넣을지 아니면 detach 할지 고민해보기
    """
    x_for_grad = x
    x_p_for_grad = x_prime.detach()

    x_mean = x_for_grad.mean(dim=(2, 3), keepdim=True)
    x_std = x_for_grad.std(dim=(2, 3), keepdim=True, unbiased=False)

    xp_mean = x_p_for_grad.mean(dim=(2, 3), keepdim=True)
    xp_std = x_p_for_grad.std(dim=(2, 3), keepdim=True, unbiased=False)

    y = (x_prime - xp_mean) / (xp_std + eps) * x_std + x_mean
    return y


class SignPow(nn.Module):
    def __init__(self, num_channels, init_alpha=0.0, eps=1e-6):
        super().__init__()
        # 채널별 alpha 파라미터
        self.raw_alpha = nn.Parameter(torch.full((num_channels,), init_alpha))
        self.eps = eps
    def forward(self, x):
        # alpha 범위: 0.9 ~ 1.1
        alpha = 1 + torch.tanh(self.raw_alpha)

        # x가 (N, C, H, W)라고 가정
        alpha = alpha.view(1, -1, 1, 1)
        x_prime = x.detach()  # x의 연산 그래프를 끊어서 x_prime으로 사용

        x =  alpha * ( torch.sign(x) * (torch.abs(x) + self.eps) ** alpha)
        return scaling_func(x, x_prime)

"""
class SignPow(nn.Module):
    def __init__(self, num_channels, init_alpha=0.0, eps=1e-6):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.full((num_channels,), init_alpha))
        self.eps = eps

    def forward(self, x):
        # alpha 범위: 0.9 ~ 1.1
        alpha = 1 + torch.tanh(self.raw_alpha) * 0.1
        alpha = alpha.view(1, -1, 1, 1)

        y = torch.sign(x) * (torch.abs(x)) ** alpha

        # forward 값은 그대로 유지
        # 단, x==0 위치는 backward만 막아서 grad=0
        mask = (x != 0).to(x.dtype)
        y = y * mask + y.detach() * (1 - mask)

        return y
"""
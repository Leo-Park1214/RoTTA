import torch
import torch.nn as nn
import torch.nn.functional as F

def scaling_func(x, x_prime, eps=1e-6):
    """
    x, x_prime: shape = (B, C, H, W)

    각 (B, C)별로 H, W 축에 대해서만 min/max를 구해서
    x_prime을 x와 같은 min/max 범위로 affine scaling
    x, xprime 을 연산 그래프에 넣을지 아니면 detach 할지 고민해보기
    
    """
    x_p_for_grad = x_prime.detach()  # x_prime의 연산 그래프를 끊어서 x_for_grad로 사용
    
    x_min = x.amin(dim=(2, 3), keepdim=True)       # (B, C, 1, 1)
    x_max = x.amax(dim=(2, 3), keepdim=True)       # (B, C, 1, 1)

    xp_min = x_p_for_grad.amin(dim=(2, 3), keepdim=True)
    xp_max = x_p_for_grad.amax(dim=(2, 3), keepdim=True)

    x_range = x_max - x_min
    xp_range = xp_max - xp_min

    scale = x_range / (xp_range + eps)

    y = (x_prime - xp_min) * scale + x_min
    return y

class SignPow(nn.Module):
    def __init__(self, init_alpha=0.0, eps=1e-6):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor([init_alpha]))
        self.eps = eps

    def forward(self, x):
        alpha = 1 + F.tanh(self.raw_alpha) * 0.1  # alpha > 0 보장)
        
        #print(alpha)
        return torch.sign(x) * (torch.abs(x) + self.eps) ** alpha
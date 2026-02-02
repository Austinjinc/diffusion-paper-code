import torch
import torch.nn as nn
import torch.nn.functional as F

class ArbitragePenalties(nn.Module):
    """
    Calculates arbitrage penalty losses.
    1. Smile: Enforces convexity of the IV smile (Heuristic for Butterfly).
    2. TTM: Enforces non-decreasing Total Variance (Calendar).
    """
    def __init__(self, device, taus):
        super().__init__()
        self.register_buffer('taus', torch.tensor(taus, dtype=torch.float32, device=device).view(1, 1, 1, -1))
        smile_kernel = torch.tensor([[[[1],[-2],[1]]]], dtype=torch.float32, device=device)
        self.smile_conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=0, bias=False)
        self.smile_conv.weight.data = smile_kernel
        self.smile_conv.weight.requires_grad = False

        ttm_kernel = torch.tensor([[[[1, -1]]]], dtype=torch.float32, device=device)
        self.ttm_conv = nn.Conv2d(1, 1, kernel_size=(1, 2), padding=0, bias=False)
        self.ttm_conv.weight.data = ttm_kernel
        self.ttm_conv.weight.requires_grad = False

    def forward(self, surface_iv_space):
        smile_curvature = self.smile_conv(surface_iv_space)
        smile_loss = torch.mean(F.relu(-smile_curvature)**2, dim=(2, 3)).squeeze()

        total_variance = (surface_iv_space ** 2) * self.taus
        ttm_diff = self.ttm_conv(total_variance)
        ttm_loss = torch.mean(F.relu(ttm_diff)**2, dim=(2, 3)).squeeze()
        
        return smile_loss, ttm_loss

class JaggednessPenalties(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Define the Laplacian kernel
        laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=device)
        self.laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
        self.laplacian_conv.weight.data = laplacian_kernel
        # Freeze the weights
        self.laplacian_conv.weight.requires_grad = False

    def forward(self, surface):
        # The penalty is the mean of the squared second-order derivatives
        laplacian_output = self.laplacian_conv(surface)
        # jaggedness_loss = torch.mean(laplacian_output**2)
        jaggedness_loss_per_sample = torch.mean(laplacian_output**2, dim=(2, 3)).squeeze()
        return jaggedness_loss_per_sample

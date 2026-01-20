import torch
import torch.nn as nn
import torch.nn.functional as F

class ArbitragePenalties(nn.Module):
    """
    Calculates arbitrage penalty losses (smile and ttm) for a denormalized surface.
    """
    def __init__(self, device):
        super().__init__()
        # Vertical kernel for smile curvature (moneyness axis)
        smile_kernel = torch.tensor([[[[1],[-2],[1]]]], dtype=torch.float32, device=device)
        self.smile_conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=0, bias=False)
        self.smile_conv.weight.data = smile_kernel
        self.smile_conv.weight.requires_grad = False

        # Horizontal kernel for TTM slope (calendar spread)
        ttm_kernel = torch.tensor([[[[1, -1]]]], dtype=torch.float32, device=device)
        self.ttm_conv = nn.Conv2d(1, 1, kernel_size=(1, 2), padding=0, bias=False)
        self.ttm_conv.weight.data = ttm_kernel
        self.ttm_conv.weight.requires_grad = False

    def forward(self, surface_iv_space):
        # Note: Input surface must be in the original IV space (not normalized)
        smile_curvature = self.smile_conv(surface_iv_space)
        # smile_loss = torch.mean(F.relu(-smile_curvature)**2) # Penalize non-convex smiles

        ttm_slope = self.ttm_conv(surface_iv_space)
        # ttm_loss = torch.mean(F.relu(-ttm_slope)**2) # Penalize downward sloping term structure
        smile_loss_per_sample = torch.mean(F.relu(-smile_curvature)**2, dim=(2, 3)).squeeze()
        ttm_loss_per_sample = torch.mean(F.relu(-ttm_slope)**2, dim=(2, 3)).squeeze()
        
        return smile_loss_per_sample, ttm_loss_per_sample

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

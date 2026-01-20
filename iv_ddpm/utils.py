import torch
import numpy as np
import random
import os

def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)

def mk_folders(run_name):
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def denormalize_surface(normalized_surface, mean_log, std_log):
    """
    Converts a normalized surface back to its original IV scale.
    Input can be a torch.Tensor or np.array.
    """
    if isinstance(normalized_surface, torch.Tensor):
        # Convert torch tensor to numpy for calculation
        normalized_surface = normalized_surface.detach().cpu().numpy()

    # Ensure stats are numpy arrays
    if isinstance(mean_log, torch.Tensor):
        mean_log = mean_log.cpu().numpy()
    if isinstance(std_log, torch.Tensor):
        std_log = std_log.cpu().numpy()

    # De-standardize
    denormalized_log_surface = normalized_surface * std_log + mean_log
    # Exponentiate to reverse the log
    original_surface = np.exp(denormalized_log_surface)
    return original_surface

def denormalize_surface_torch(normalized_surface, mean_log, std_log):

    device = normalized_surface.device
    dtype = normalized_surface.dtype

    mean_log_torch = torch.from_numpy(mean_log).to(device, dtype=dtype)
    std_log_torch = torch.from_numpy(std_log).to(device, dtype=dtype)

    # De-standardize (y = z * sigma + mu)
    denormalized_log = normalized_surface * std_log_torch + mean_log_torch
    original_surface = torch.exp(denormalized_log)
    
    return original_surface
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler
from .financial_utils import penalty_mutau

# =================================================================================
# == 4. DATA HANDLING ============================================================
# =================================================================================

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for loading time-series data.
    """
    def __init__(self, surfaces, surfaces_ema_short, surfaces_ema_long, scalars, dates):
        self.surfaces = surfaces
        self.surfaces_ema_short = surfaces_ema_short
        self.surfaces_ema_long = surfaces_ema_long
        self.scalars = scalars
        self.dates = dates

    def __len__(self):
        return len(self.surfaces) - 1

    def __getitem__(self, idx):
        # today's surface, today's scalars, tomorrow's surface
        return (self.surfaces[idx], 
                self.surfaces_ema_short[idx], 
                self.surfaces_ema_long[idx], 
                self.scalars[idx], 
                self.surfaces[idx+1])

def get_dataloaders(cfg, is_evaluation=False):
    """
    Loads and processes data into DataLoaders.
    
    Has two modes:
    1. is_evaluation=False: (For training) Loads all data, splits it,
       calculates stats, saves stats, and returns all 3 loaders.
    2. is_evaluation=True: (For evaluation) Loads all data, loads
       pre-calculated stats, and returns only the test loader and grid info.
    """
    data_path = 'data'
    try:
        all_surfaces = np.load(os.path.join(data_path, 'iv_surfaces.npy'))
        all_scalars = np.load(os.path.join(data_path, 'cond_array.npy'))
        all_dates = np.load(os.path.join(data_path, 'dates.npy'))
        logging.info("Loaded data successfully from 'data/' folder.")
    except (FileNotFoundError, AssertionError) as e:
        logging.error(f"Error loading data files from 'data/' folder: {e}.")
        logging.error("Please run 'scripts/00_acquire_data.py' first.")
        # This was the cause of your error. We will now raise it properly.
        raise e

    # --- Common Data Processing ---
    n_days, H, W = all_surfaces.shape
    surfaces_flat = all_surfaces.reshape(n_days, H * W)
    surfaces_df = pd.DataFrame(surfaces_flat)

    surfaces_ema_short_flat = surfaces_df.ewm(span=5, adjust=False).mean().values
    all_surfaces_ema_short = surfaces_ema_short_flat.reshape(n_days, H, W)

    surfaces_ema_long_flat = surfaces_df.ewm(span=20, adjust=False).mean().values
    all_surfaces_ema_long = surfaces_ema_long_flat.reshape(n_days, H, W)
    
    train_end_idx = int(n_days * cfg.train_split)
    val_end_idx = train_end_idx + int(n_days * cfg.val_split)

    # --- Mode-Specific Logic ---
    if is_evaluation:
        # --- EVALUATION MODE ---
        # We only need to load the stats, not calculate them
        stats_path = os.path.join('models', cfg.run_name, "surface_stats.npz")
        try:
            stats_file = np.load(stats_path)
            mean_log_surface = stats_file['mean']
            std_log_surface = stats_file['std']
            norm_stats = {'mean': mean_log_surface, 'std': std_log_surface}
            logging.info(f"Loaded normalization stats from {stats_path}")
        except FileNotFoundError:
            logging.error(f"FATAL: Stats file not found at {stats_path}. Cannot run evaluation.")
            raise
    else:
        # --- TRAINING MODE ---
        # Calculate stats from the training split
        train_surfaces_raw = all_surfaces[:train_end_idx]
        log_train_surfaces = np.log(train_surfaces_raw + 1e-8)
        mean_log_surface = np.mean(log_train_surfaces, axis=0)
        std_log_surface = np.std(log_train_surfaces, axis=0)
        std_log_surface[std_log_surface == 0] = 1.0
        
        # Save the stats
        stats_path = os.path.join('models', cfg.run_name, "surface_stats.npz")
        try:
            # We need to ensure the models/run_name directory exists
            os.makedirs(os.path.join('models', cfg.run_name), exist_ok=True)
            np.savez(stats_path, mean=mean_log_surface, std=std_log_surface)
            logging.info(f"Calculated and saved normalization stats to {stats_path}")
        except FileNotFoundError:
            logging.error(f"Could not save stats. Directory 'models/{cfg.run_name}' does not exist.")
            raise
            
        norm_stats = {'mean': mean_log_surface, 'std': std_log_surface}

    # --- Common Data Transformation (Post-Stats) ---
    surfaces_normalized = (np.log(all_surfaces + 1e-8) - mean_log_surface) / std_log_surface
    surfaces_ema_short_normalized = (np.log(all_surfaces_ema_short + 1e-8) - mean_log_surface) / std_log_surface
    surfaces_ema_long_normalized = (np.log(all_surfaces_ema_long + 1e-8) - mean_log_surface) / std_log_surface
    
    all_surfaces_tensor = torch.from_numpy(surfaces_normalized).float().unsqueeze(1)
    all_surfaces_ema_short_tensor = torch.from_numpy(surfaces_ema_short_normalized).float().unsqueeze(1)
    all_surfaces_ema_long_tensor = torch.from_numpy(surfaces_ema_long_normalized).float().unsqueeze(1)

    # --- Scaler: Fit on train, transform all ---
    train_scalars_np = all_scalars[:train_end_idx]
    scaler = StandardScaler()
    scaler.fit(train_scalars_np)
    
    # Scale all splits
    train_scalars_scaled_np = scaler.transform(train_scalars_np)
    val_scalars_scaled_np = scaler.transform(all_scalars[train_end_idx:val_end_idx])
    test_scalars_scaled_np = scaler.transform(all_scalars[val_end_idx:])
    
    train_scalars_tensor = torch.from_numpy(train_scalars_scaled_np).float()
    val_scalars_tensor = torch.from_numpy(val_scalars_scaled_np).float()
    test_scalars_tensor = torch.from_numpy(test_scalars_scaled_np).float()

    # --- Create Datasets & Loaders ---
    
    # Grid info is always needed
    m_grid = np.linspace(0.6, 1.4, 9)
    ttm_grid = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1])
    P_T, P_K, PB_K = penalty_mutau(m_grid, ttm_grid)

    # Create the test dataset (always needed)
    test_dataset = TimeSeriesDataset(
        all_surfaces_tensor[val_end_idx:],
        all_surfaces_ema_short_tensor[val_end_idx:],
        all_surfaces_ema_long_tensor[val_end_idx:],
        test_scalars_tensor,
        all_dates[val_end_idx:]
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    if is_evaluation:
        # For evaluation, we only return what's needed
        return None, None, test_loader, norm_stats, m_grid, ttm_grid, P_T, P_K, PB_K
    
    # --- For Training Mode, create train/val datasets ---
    train_dataset = TimeSeriesDataset(
        all_surfaces_tensor[:train_end_idx],
        all_surfaces_ema_short_tensor[:train_end_idx],
        all_surfaces_ema_long_tensor[:train_end_idx],
        train_scalars_tensor,
        all_dates[:train_end_idx]
    )
    val_dataset = TimeSeriesDataset(
        all_surfaces_tensor[train_end_idx:val_end_idx],
        all_surfaces_ema_short_tensor[train_end_idx:val_end_idx],
        all_surfaces_ema_long_tensor[train_end_idx:val_end_idx],
        val_scalars_tensor,
        all_dates[train_end_idx:val_end_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, norm_stats, m_grid, ttm_grid, P_T, P_K, PB_K
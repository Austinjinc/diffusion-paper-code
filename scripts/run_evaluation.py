import torch
import logging
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
from datetime import datetime

# --- IMPORTANT ---
# Add this block to import from your package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports from your 'iv_ddpm' package
from iv_ddpm.data_loader import get_dataloaders
from iv_ddpm.model import ConditionalUnet
from iv_ddpm.diffusion import Trainer, get_cosine_schedule
from iv_ddpm.financial_utils import calculate_penalties_vectorized, penalty_mutau
from iv_ddpm.plotting import (
    plot_iv_slice_comparison_with_ci, 
    plot_arbitrage_timeseries_with_ci, 
    plot_normalized_deviation,
    plot_mape_timeseries_with_ci,
    plot_mean_surface_grid
)
from iv_ddpm.metrics import calculate_mape_per_sample_day, calculate_normalized_deviation
from iv_ddpm.utils import denormalize_surface

# We also need get_config from the training script
try:
    from run_hyper_search import get_config
except ImportError:
    print("ERROR: Could not import 'get_config' from 'run_hyper_search.py'.")
    print("Please ensure 'run_hyper_search.py' is in the same 'scripts/' folder.")
    sys.exit(1)

# =================================================================================
# == EVALUATION & SAMPLING FUNCTIONS (from original DDPM.py) ======================
# =================================================================================
# (Note: This function is copied from the original file for this script to use)
def run_evaluation(cfg, test_loader, m_grid, ttm_grid, P_T, P_K, PB_K, norm_stats):
    """
    Loads the best model and runs a full evaluation on the test set.
    """
    logging.info(f"--- Starting Evaluation Phase for run: {cfg.run_name} ---")
    device = cfg.device

    # --- Load Normalization Stats ---
    if not norm_stats:
        stats_path = os.path.join('models', cfg.run_name, "surface_stats.npz")
        if not os.path.exists(stats_path):
            logging.error(f"Stats file not found at {stats_path}. Please run training first.")
            return
        logging.info(f"Loading stats from {stats_path}")
        stats_file = np.load(stats_path)
        norm_stats = {'mean': stats_file['mean'], 'std': stats_file['std']}

    # --- Load Best Model ---
    model = ConditionalUnet(cfg).to(device)
    loss_model_path = os.path.join("models", cfg.run_name, "best_loss_model.pt")
    model_to_load = None
    if os.path.exists(loss_model_path):
        model_to_load = loss_model_path
        logging.info(f"Loading model from {loss_model_path}")
    else:
        logging.error(f"Evaluation failed: No model file was found at {loss_model_path}.")
        return

    model.load_state_dict(torch.load(model_to_load, map_location=device))

    # --- Recreate Diffusion Schedule ---
    beta = get_cosine_schedule(cfg.noise_steps).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    eval_trainer = Trainer(model, train_loader=None, val_loader=None, cfg=cfg)
    all_samples_list = []
    
    logging.info(f"Generating {cfg.n_samples} samples for each day in the test set...")
    for i in range(cfg.n_samples):
        all_predicted_s_one_run = []
        model.eval()
        pbar_desc = f"Generating Sample Set {i+1}/{cfg.n_samples}"
        for cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, _ in tqdm(test_loader, desc=pbar_desc, disable="SLURM_JOB_ID" in os.environ):
            cond_s = cond_s.to(device)
            cond_ema_s_short = cond_ema_s_short.to(device)
            cond_ema_s_long = cond_ema_s_long.to(device)
            cond_sc = cond_sc.to(device)
            predicted_s = eval_trainer.sample(cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc)
            all_predicted_s_one_run.append(predicted_s.cpu())
        all_samples_list.append(torch.cat(all_predicted_s_one_run, dim=0))
    
    # Stack samples: shape will be (n_samples, n_days, c, h, w)
    all_samples_tensor = torch.stack(all_samples_list, dim=0)
    all_target_s = torch.cat([target_s for _, _, _, _, target_s in test_loader], dim=0)
    final_dates = test_loader.dataset.dates[1:]

    true_surfaces_denorm = denormalize_surface(all_target_s.squeeze(1), norm_stats['mean'], norm_stats['std'])
    all_samples_denorm = denormalize_surface(all_samples_tensor.squeeze(2), norm_stats['mean'], norm_stats['std'])
    
    mean_predictions = np.mean(all_samples_denorm, axis=0)
    lower_bound_preds = np.percentile(all_samples_denorm, 5, axis=0)  # 5th percentile for 90% CI
    upper_bound_preds = np.percentile(all_samples_denorm, 95, axis=0) # 95th percentile for 90% CI
    
    results_dir = os.path.join("results", cfg.run_name)
    save_path_numpy = os.path.join(results_dir, "test_set_predictions.npy")
    np.save(save_path_numpy, all_samples_denorm)
    logging.info(f"Saved test set predictions to {save_path_numpy}")
    
    # Arbitrage Penalty Calculation
    logging.info("Calculating arbitrage penalties...")
    n_samples, n_days, H, W = all_samples_denorm.shape
    all_samples_flat = all_samples_denorm.reshape(n_samples * n_days, H, W)
    _, _, _, all_penalties_flat = calculate_penalties_vectorized(all_samples_flat, m_grid, ttm_grid, P_T, P_K, PB_K)
    _, _, _, true_penalties = calculate_penalties_vectorized(true_surfaces_denorm, m_grid, ttm_grid, P_T, P_K, PB_K)
    all_penalties = all_penalties_flat.reshape(n_samples, n_days)
    lower_bound_arbitrage = np.percentile(all_penalties, 5, axis=0)
    mean_penalties = np.mean(all_penalties, axis=0)
    upper_bound_arbitrage = np.percentile(all_penalties, 95, axis=0)
    plot_arbitrage_timeseries_with_ci(
        dates=final_dates,
        true_penalties=true_penalties,
        lower_bound=lower_bound_arbitrage,
        mean=mean_penalties,
        upper_bound=upper_bound_arbitrage,
        run_name=cfg.run_name
    )
    
    # --- START: Refactored Plotting Section ---
    logging.info("--- Generating comparison and deviation plots for specific IV slices ---")

    ATM_M_IDX = 4
    ITM_M_IDX = 1 # Corresponds to moneyness of ~0.7
    OTM_M_IDX = 7 # Corresponds to moneyness of ~1.3
    ONE_DAY_T_IDX = 0
    ONE_WEEK_T_IDX = 1
    ONE_MONTH_T_IDX = 3
    THREE_MONTH_T_IDX = 5
    slices_to_plot = [
        {"moneyness_idx": ATM_M_IDX, "ttm_idx": ONE_DAY_T_IDX,     "name": "ATM 1-Day"},
        {"moneyness_idx": ATM_M_IDX, "ttm_idx": ONE_WEEK_T_IDX,    "name": "ATM 1-Week"},
        {"moneyness_idx": ATM_M_IDX, "ttm_idx": ONE_MONTH_T_IDX,   "name": "ATM 1-Month"},
        {"moneyness_idx": ATM_M_IDX, "ttm_idx": THREE_MONTH_T_IDX, "name": "ATM 3-Month"},
        {"moneyness_idx": ITM_M_IDX, "ttm_idx": ONE_WEEK_T_IDX,     "name": "ITM 1-Week"},
        {"moneyness_idx": ITM_M_IDX, "ttm_idx": ONE_MONTH_T_IDX,   "name": "ITM 1-Month"},
        {"moneyness_idx": ITM_M_IDX, "ttm_idx": THREE_MONTH_T_IDX, "name": "ITM 3-Month"},
        {"moneyness_idx": OTM_M_IDX, "ttm_idx": ONE_WEEK_T_IDX,     "name": "OTM 1-Week"},
        {"moneyness_idx": OTM_M_IDX, "ttm_idx": ONE_MONTH_T_IDX,   "name": "OTM 1-Month"},
        {"moneyness_idx": OTM_M_IDX, "ttm_idx": THREE_MONTH_T_IDX, "name": "OTM 3-Month"},
    ]

    for s in slices_to_plot:
        m_idx, t_idx, name = s["moneyness_idx"], s["ttm_idx"], s["name"]

        plot_iv_slice_comparison_with_ci(
            dates=final_dates,
            true_surfaces=true_surfaces_denorm,
            mean_predicted_surfaces=mean_predictions,
            lower_bound_surfaces=lower_bound_preds,
            upper_bound_surfaces=upper_bound_preds,
            moneyness_idx=m_idx,
            ttm_idx=t_idx,
            title=f"{name} Implied Volatility with 90% Confidence Interval",
            run_name=cfg.run_name
        )

        true_slice = true_surfaces_denorm[:, m_idx, t_idx]
        lower_slice = lower_bound_preds[:, m_idx, t_idx]
        upper_slice = upper_bound_preds[:, m_idx, t_idx]
        
        deviation_scores = calculate_normalized_deviation(true_slice, lower_slice, upper_slice)
        
        # Finally, create the plot
        plot_normalized_deviation(
            dates=final_dates,
            scores=deviation_scores,
            title=f"Normalized Deviation for {name} IV",
            run_name=cfg.run_name
        )

    logging.info("Calculating MAPE time series...")
    daily_mape_per_sample = calculate_mape_per_sample_day(all_samples_denorm, true_surfaces_denorm)
    median_mape = np.percentile(daily_mape_per_sample, 50, axis=0)
    lower_bound_mape = np.percentile(daily_mape_per_sample, 5, axis=0)
    upper_bound_mape = np.percentile(daily_mape_per_sample, 95, axis=0)
    plot_mape_timeseries_with_ci(
        dates=final_dates,
        median_mape=median_mape,
        lower_bound=lower_bound_mape,
        upper_bound=upper_bound_mape,
        run_name=cfg.run_name
    )
    
    logging.info("Plotting mean surface grid...")
    plot_mean_surface_grid(
        surfaces_to_plot=mean_predictions,
        dates_to_plot=final_dates,
        run_name=cfg.run_name,
        n_plots=20
    )
    
    logging.info("--- Evaluation Complete ---")

# =================================================================================
# == NEW HELPER FUNCTION TO FIND LATEST RUN =======================================
# =================================================================================

def find_most_recent_run(base_run_name, models_dir="models"):
    """
    Finds the most recent run folder in the models directory that matches
    the base_run_name.
    """
    logging.info(f"Searching for runs starting with '{base_run_name}' in '{models_dir}/'")
    
    # Create the search pattern
    # e.g., models/W_mse_log_iv_norm_per_cell*
    search_pattern = os.path.join(models_dir, f"{base_run_name}*")
    
    # Use glob to find all directories matching the pattern
    matching_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    
    if not matching_dirs:
        logging.warning(f"No matching directories found for pattern: {search_pattern}")
        return None
    
    # Sort the directories. Since the timestamp is YYYY-MM-DD_HH-MM,
    # a simple string sort will find the most recent.
    matching_dirs.sort()
    
    # Get the latest one (the last in the sorted list)
    latest_run_dir = matching_dirs[-1]
    
    # Return just the folder name, not the full path
    latest_run_name = os.path.basename(latest_run_dir)
    return latest_run_name

# =================================================================================
# == SCRIPT EXECUTION =============================================================
# =================================================================================

if __name__ == '__main__':
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        force=True
    )

    cfg = get_config()

    # --- START: NEW LOGIC TO FIND CORRECT RUN NAME ---
    # Find the latest timestamped run folder based on the default config name
    latest_run_name = find_most_recent_run(cfg.run_name, models_dir="models")
    
    if latest_run_name:
        logging.info(f"Found most recent run folder: {latest_run_name}")
        # Overwrite the default config name with the full, correct name
        cfg.run_name = latest_run_name
    else:
        logging.warning(f"Could not find a matching run folder for base '{cfg.run_name}'.")
        logging.warning(f"Will attempt to load from default path: 'models/{cfg.run_name}'")
    # --- END: NEW LOGIC ---

    # --- Data Loading ---
    # We load the data here to get the test_loader and grid info
    logging.info("Loading test data...")
    try:
        # Note: We pass cfg here, but data_loader doesn't use cfg.run_name
        _, _, test_loader, norm_stats, m_grid, ttm_grid, P_T, P_K, PB_K = get_dataloaders(cfg, is_evaluation=True)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        logging.error("Please ensure your 'data/' folder is populated by running 'scripts/00_acquire_data.py' first.")
        sys.exit(1)

    # --- Run Evaluation ---
    # The 'cfg' object now contains the *correct, full* run_name
    # (e.g., W_mse_log_iv_norm_per_cell_2025-11-07_23-54_joblocal)
    run_evaluation(cfg, test_loader, m_grid, ttm_grid, P_T, P_K, PB_K, norm_stats)
"""
Main script for running the Optuna hyperparameter search.

This script:
1.  Sets up the configuration (`get_config`).
2.  Initializes logging.
3.  Loads and prepares the data.
4.  Defines the Optuna `objective` function.
5.  Starts the Optuna study.
"""

import torch
import logging
import os
import argparse
import optuna
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports from our 'iv_ddpm' package
from iv_ddpm.data_loader import get_dataloaders
from iv_ddpm.model import ConditionalUnet
from iv_ddpm.diffusion import Trainer, get_cosine_schedule
from iv_ddpm.metrics import calculate_quality_metrics
from iv_ddpm.utils import set_seed, mk_folders
from iv_ddpm.plotting import plot_loss_curves, plot_gradient_norms, plot_lr_curve
from iv_ddpm.financial_utils import penalty_mutau

# =================================================================================
#  CONFIGURATION
# =================================================================================
def get_config():
    parser = argparse.ArgumentParser(description="Train a Diffusion Model for IV Surface Forecasting")

    # -- Run Details --
    parser.add_argument('--run_name', type=str, default="W_mse_log_iv_norm_per_cell", help='A name for this training run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for training')

    # -- Data & Model Shape --
    parser.add_argument('--c_in', type=int, default=4, help='Input channels (today_surface + noised_target)')
    parser.add_argument('--c_out', type=int, default=1, help='Output channels (predicted noise)')
    parser.add_argument('--n_scalars', type=int, default=5, help='Number of scalar condition features')
    parser.add_argument('--enc_channels', type=int, default=16, help='Channels in the first level of the U-Net')
    parser.add_argument('--bottle_channels', type=int, default=30, help='Channels in the bottleneck of the U-Net')
    parser.add_argument('--emb_dim', type=int, default=20, help='Dimension of the embedding')

    # -- Diffusion Process --
    parser.add_argument('--noise_steps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--EMA_beta', type=float, default=0.995, help='Decay rate for EMA of model parameters')
    parser.add_argument('--sampler', type=str, default="ddpm", choices=['ddim', 'ddpm'], help='Sampler to use (ddim or ddpm)')
    # -- Training --
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--val_split', type=float, default=0.1, help='Proportion of data for validation')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of Optuna trials for hyperparameter tuning')
    parser.add_argument('--arbitrage_threshold', type=float, default=0.005, help='Threshold for arbitrage penalty to consider a surface valid')
    # -- Monitoring --
    parser.add_argument('--plot_every', type=int, default=3, help='Plot sample surfaces every N epochs')
    parser.add_argument('--n_samples', type=int, default=2, help='Number of surfaces to sample per input for evaluation')

    # Parse args
    args = parser.parse_args()
    return args

# =================================================================================
# OPTUNA OBJECTIVE
# =================================================================================
def objective(trial, cfg, train_loader, val_loader, norm_stats, arbitrage_threshold, mape_tolerance, study_logger):
    # Suggest hyperparameters for Optuna to tune
    # cfg.enc_channels = trial.suggest_categorical("enc_channels", [4, 6, 8])
    # cfg.bottle_channels = cfg.enc_channels * 2

    # Initialize model with the suggested hyperparameters
    model = ConditionalUnet(cfg)
    m_grid = np.linspace(0.6, 1.4, 9) # Re-define grids for the function
    ttm_grid = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1])
    trainer = Trainer(model, train_loader, val_loader, cfg, trial, norm_stats, m_grid, ttm_grid, arbitrage_threshold=cfg.arbitrage_threshold)
    
    study_logger.info(f"--- Starting training with trial {trial.number} and parameters: {trial.params} ---")
    trainer.fit()
    
    best_model_path = os.path.join("models", cfg.run_name, "best_loss_model.pt")
    if not os.path.exists(best_model_path):
        study_logger.info(f"Best model not found at {best_model_path}. Starting training.")
        return float('inf')  # Return infinity to mark as a failure if the model is not trained
    eval_model = ConditionalUnet(cfg).to(cfg.device)
    eval_model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    
    all_target_s, all_predicted_s = [], []
    eval_model.eval()
    with torch.no_grad():
        for cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, target_s in val_loader:
            predicted_s = trainer.sample(
                        cond_s.to(cfg.device),
                        cond_ema_s_short.to(cfg.device),
                        cond_ema_s_long.to(cfg.device),
                        cond_sc.to(cfg.device)
            )
            
            all_target_s.append(target_s)
            all_predicted_s.append(predicted_s.cpu())

    final_target_s = torch.cat(all_target_s, dim=0).squeeze(1)
    final_predicted_s = torch.cat(all_predicted_s, dim=0).squeeze(1)

    # Calculate the quality metrics for the generated surfaces
    generated_metrics = calculate_quality_metrics(
        final_predicted_s, final_target_s, norm_stats, m_grid, ttm_grid
    )
    study_logger.info(f"--- [Trial {trial.number}] Generated Metrics: {generated_metrics} ---")
    trial.set_user_attr("arbitrage", generated_metrics["arbitrage"])
    trial.set_user_attr("accuracy_mape", generated_metrics["accuracy_mape"])
    
    if generated_metrics["arbitrage"] > arbitrage_threshold:
        study_logger.info(f"--- [Trial {trial.number}] FAILED: Arbitrage too high. ---")
        return float('inf')

    if generated_metrics["accuracy_mape"] > mape_tolerance:
        study_logger.info(f"--- [Trial {trial.number}] FAILED: MAPE exceeds tolerance of {mape_tolerance}%. ---")
        return float('inf')
    
    study_logger.info(f"--- [Trial {trial.number}] PASSED Quality Gauntlet. Final Score (MAPE): {generated_metrics['accuracy_mape']:.6f} ---")

    return generated_metrics["accuracy_mape"]


# =================================================================================
# SCRIPT EXECUTION
# =================================================================================
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    
    cfg = get_config()

    # Folder name config here
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local") # Use 'local' if not on a Slurm cluster
    cfg.run_name = f"{cfg.run_name}_{timestamp}_job{job_id}"

    set_seed(cfg.seed)
    mk_folders(cfg.run_name)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        force=True
    )

    study_logger = logging.getLogger('study_logger')
    study_logger.setLevel(logging.INFO)
    log_file_path = os.path.join("results", cfg.run_name, "study_summary.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
    study_logger.addHandler(file_handler)
    study_logger.propagate = False
    logging.info(f"Optuna study logs will be saved to: {log_file_path}")

    # --- Data Loading (now from data_loader.py) ---
    logging.info("Loading and processing data...")
    train_loader, val_loader, test_loader, norm_stats = get_dataloaders(cfg)
    logging.info("Data loading complete.")

    # --- Run Training ---
    db_path = os.path.join("results", cfg.run_name, "iv_surface_study.db")
    
    m_grid = np.linspace(0.6, 1.4, 9)
    ttm_grid = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1])
    P_T, P_K, PB_K = penalty_mutau(m_grid, ttm_grid)
    
    # Get stats from the val_dataset (which is inside the loader)
    real_val_surfaces_norm = val_loader.dataset.surfaces[:, 0, :, :]
    real_metrics = calculate_quality_metrics(
        predicted_surfaces_norm=real_val_surfaces_norm,
        real_surfaces_norm=real_val_surfaces_norm,
        norm_stats=norm_stats,
        m_grid=m_grid,
        ttm_grid=ttm_grid
    )
    arbitrage_threshold = real_metrics['arbitrage'] + cfg.arbitrage_threshold  # Set a threshold
    study_logger.info(f"Arbitrage threshold set to: {arbitrage_threshold:.4f}")
    
    successful_trials = []
    def count_successful_trials_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.value != float('inf'):
                successful_trials.append(trial)

    study = optuna.create_study(
        study_name="iv_surface_tuning",
        storage=f"sqlite:///{db_path}",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )
    mape_tolerance = 5.0  # Set a tolerance for MAPE, e.g., 5%
    study.optimize(
        lambda trial: objective(trial, cfg, train_loader, val_loader, norm_stats, arbitrage_threshold, mape_tolerance, study_logger),
        n_trials=cfg.n_trials,
        callbacks=[count_successful_trials_callback]
    )

    num_successful = len(successful_trials)
    study_logger.info(f"Number of trials that passed the Quality Gauntlet: {num_successful} out of {len(study.trials)}")

    # (The rest of your __main__ block for logging and reporting)
    if num_successful > 0:
        successful_trial_numbers = [t.number for t in successful_trials]
        study_logger.info(f"Successful Trial Numbers: {successful_trial_numbers}")
        study_logger.info(f"--- Best Successful Trial (Trial #{study.best_trial.number}) ---")
        study_logger.info(f"Best trial value: {study.best_trial.value:.6f}")
        study_logger.info(f"Best trial parameters: {study.best_trial.params}")
        if 'arbitrage' in study.best_trial.user_attrs:
            best_trial_metrics = study.best_trial.user_attrs
            study_logger.info(f"Metrics: Arbitrage={best_trial_metrics['arbitrage']:.4f}, Accuracy (MAPE)={best_trial_metrics['accuracy_mape']:.2f}%")
    else:
        study_logger.warning("No trials passed the quality gauntlet.")
        
        evaluated_failures = []
        for trial in study.trials:
            if "arbitrage" in trial.user_attrs and "accuracy_mape" in trial.user_attrs:
                metrics = trial.user_attrs
                if not any(np.isnan(v) for v in [metrics["arbitrage"], metrics["accuracy_mape"]]):
                    evaluated_failures.append({
                        "trial_number": trial.number,
                        "params": trial.params,
                        "arbitrage": metrics["arbitrage"],
                        "accuracy_mape": metrics["accuracy_mape"]
                    })

        if evaluated_failures:
            failed_df = pd.DataFrame(evaluated_failures)
            failed_df['arbitrage_rank'] = failed_df['arbitrage'].rank()
            failed_df['mape_rank'] = failed_df['accuracy_mape'].rank()
            failed_df['total_rank'] = failed_df['arbitrage_rank'] + failed_df['mape_rank']

            best_failed_trial_info = failed_df.loc[failed_df['total_rank'].idxmin()]

            study_logger.info("-" * 60)
            study_logger.info(f"--- Details of the 'Best Failed Trial' (Trial #{int(best_failed_trial_info['trial_number'])}) ---")
            study_logger.info("This trial had the best overall rank for arbitrage and MAPE.")
            study_logger.info(f"Params: {best_failed_trial_info['params']}")
            study_logger.info(f"Arbitrage Score: {best_failed_trial_info['arbitrage']:.4f} (Threshold was {arbitrage_threshold:.4f})")
            study_logger.info(f"Accuracy (MAPE): {best_failed_trial_info['accuracy_mape']:.2f}% (Threshold was {mape_tolerance}%)")

        else:
            study_logger.warning("Could not determine a best failed trial. All trials may have crashed before evaluation.")
    
    final_report_trial = None
    if num_successful > 0:
        final_report_trial = study.best_trial
    elif 'best_failed_trial_info' in locals():
        final_report_trial_number = int(best_failed_trial_info['trial_number'])
        final_report_trial = study.trials[final_report_trial_number]

    if final_report_trial:
        logging.info(f"Copying plots from best trial (Trial {final_report_trial.number}) for final report.")
        source_dir = os.path.join("results", cfg.run_name, f"trial_{final_report_trial.number}")
        dest_dir = os.path.join("results", cfg.run_name)
        files_to_copy = {
            "loss_curves.png": "BEST_TRIAL_loss_curves.png",
            "gradient_norms.png": "BEST_TRIAL_gradient_norms.png",
            "lr_curve.png": "BEST_TRIAL_lr_curve.png",
        }
        for original_name, new_name in files_to_copy.items():
            source_path = os.path.join(source_dir, original_name)
            dest_path = os.path.join(dest_dir, new_name)
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                logging.info(f" -> Copied {new_name}")
            else:
                logging.warning(f" -> Could not find {original_name} in {source_dir} to copy.")
    
    logging.info("Hyperparameter search complete. Run 'run_evaluation.py' for final test set evaluation.")
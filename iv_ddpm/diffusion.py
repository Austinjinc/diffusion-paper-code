import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import os
import random
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import optuna

# Relative imports
from .losses import ArbitragePenalties
from .utils import denormalize_surface_torch
from .metrics import calculate_quality_metrics
from .plotting import plot_surfaces, plot_loss_curves, plot_gradient_norms, plot_lr_curve

def get_cosine_schedule(noise_steps, s=0.008):
    """Generates a cosine beta schedule as proposed in 'Improved DDPM'."""
    steps = noise_steps + 1
    x = torch.linspace(0, noise_steps, steps)
    alphas_cumprod = torch.cos(((x / noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_linear_schedule(noise_steps, beta_start=1e-4, beta_end=0.02):
    """Generates a linear beta schedule."""
    return torch.linspace(beta_start, beta_end, noise_steps)


class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, new_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, new_weight)

    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, trial=None, norm_stats=None, m_grid=None, ttm_grid=None, arbitrage_threshold=None):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = cfg.device
        self.trial = trial

        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema = EMA(beta=cfg.EMA_beta)

        self.lr_history = []
        # In a real Optuna study, these would be suggested by the trial
        self.lr = trial.suggest_float("lr", 1e-5, 9e-4, log=True) if trial else cfg.lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=300, min_lr=1e-6
        )

        self.norm_stats = norm_stats
        if self.norm_stats:
            self.mean_log_torch = torch.from_numpy(norm_stats['mean']).to(self.device)
            self.std_log_torch = torch.from_numpy(norm_stats['std']).to(self.device)
            
        self.penalty_fn = ArbitragePenalties(self.device, taus=ttm_grid)
        self.lambda_smile = trial.suggest_float("lambda_smile", 1.4, 15, log=True) if trial else 0.01
        self.lambda_ttm = trial.suggest_float("lambda_ttm", 1, 15, log=True) if trial else 0.01

        self.beta = get_cosine_schedule(cfg.noise_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.m_grid = m_grid
        self.ttm_grid = ttm_grid
        self.arbitrage_threshold = arbitrage_threshold
    
    def evaluate_metrics(self):
        """Runs predictions on the validation set and returns metrics."""
        self.ema_model.eval()
        all_target_s, all_predicted_s = [], []
        with torch.no_grad():
            for cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, target_s in self.val_loader:
                cond_s = cond_s.to(self.device)
                cond_ema_s_short = cond_ema_s_short.to(self.device)
                cond_ema_s_long = cond_ema_s_long.to(self.device)
                cond_sc = cond_sc.to(self.device)

                # Use the stable sampling method with the EMA model
                predicted_s = self.sample(cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc)

                all_target_s.append(target_s.cpu())
                all_predicted_s.append(predicted_s.cpu())

        final_target_s = torch.cat(all_target_s, dim=0)
        final_predicted_s = torch.cat(all_predicted_s, dim=0)

        # Use existing utility function to calculate the metrics
        metrics = calculate_quality_metrics(
            final_predicted_s, final_target_s, self.norm_stats, self.m_grid, self.ttm_grid
        )
        return metrics

    def noise_surfaces(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        noised_surface = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
        v_target = sqrt_alpha_hat * noise - sqrt_one_minus_alpha_hat * x0
        return noised_surface, v_target

    def sample(self, cond_surface, cond_ema_short, cond_ema_long, scalars, n_steps=100):
        if self.cfg.sampler == 'ddim':
            # Use the fast DDIM sampler
            return self._sample_ddim(cond_surface, cond_ema_short, cond_ema_long, scalars, n_steps=n_steps)
        elif self.cfg.sampler == 'ddpm':
            # Use the full-step DDPM sampler
            return self._sample_ddpm(cond_surface, cond_ema_short, cond_ema_long, scalars)
        else:
            raise ValueError(f"Unknown sampler: {self.cfg.sampler}")
        
    def _sample_ddpm(self, cond_surface, cond_ema_short, cond_ema_long, scalars):
        """
        Stable sampling method using the EMA model and x0 clipping.
        """
        n = cond_surface.shape[0]
        self.ema_model.eval() # Use the EMA model for sampling
        device = self.device
        with torch.no_grad():
            x_t = torch.randn((n, self.cfg.c_out, 9, 9), device=device)
            # The loop is the same
            for i in tqdm(reversed(range(1, self.cfg.noise_steps)), leave=False, desc="Stable Sampling", disable="SLURM_JOB_ID" in os.environ):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                
                # Predict noise using the EMA model
                model_input = torch.cat([cond_surface, cond_ema_short, cond_ema_long, x_t], dim=1)
                predicted_v = self.ema_model(model_input, t, scalars)

                alpha_t = self.alpha[t][:, None, None, None]
                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                beta_t = self.beta[t][:, None, None, None]
                sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
                sqrt_one_minus_alpha_hat = torch.sqrt(1. - alpha_hat_t)
                alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None]
                
                pred_x0 = sqrt_alpha_hat * x_t - sqrt_one_minus_alpha_hat * predicted_v
                # pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)
                
                mean_pred = (torch.sqrt(alpha_hat_prev) * beta_t / (1. - alpha_hat_t)) * pred_x0 + \
                            (torch.sqrt(alpha_t) * (1. - alpha_hat_prev) / (1. - alpha_hat_t)) * x_t

                noise = torch.randn_like(x_t) if i > 1 else torch.zeros_like(x_t)
                posterior_variance = (beta_t * (1. - alpha_hat_prev)) / (1. - alpha_hat_t)
                x_t = mean_pred + torch.sqrt(posterior_variance) * noise

        return x_t
    
    def _sample_ddim(self, cond_surface, cond_ema_short, cond_ema_long, scalars, n_steps=100, eta=0.5):

        n = cond_surface.shape[0]
        self.ema_model.eval()
        device = self.device

        times = torch.linspace(self.cfg.noise_steps - 1, 0, n_steps + 1, device=device)
        times = list(times.int().tolist())
        time_pairs = list(zip(times[:-1], times[1:])) 

        with torch.no_grad():
            x_t = torch.randn((n, self.cfg.c_out, 9, 9), device=device)

            for time, time_next in tqdm(time_pairs, leave=False, desc="DDIM Sampling"):
                t = torch.full((n,), time, device=device, dtype=torch.long)
                t_next = torch.full((n,), time_next, device=device, dtype=torch.long)

                model_input = torch.cat([cond_surface, cond_ema_short, cond_ema_long, x_t], dim=1)
                predicted_v = self.ema_model(model_input, t, scalars)

                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                alpha_hat_next = self.alpha_hat[t_next][:, None, None, None]

                sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
                sqrt_one_minus_alpha_hat_t = torch.sqrt(1. - alpha_hat_t)

                pred_x0 = sqrt_alpha_hat_t * x_t - sqrt_one_minus_alpha_hat_t * predicted_v
                pred_epsilon = sqrt_alpha_hat_t * predicted_v + sqrt_one_minus_alpha_hat_t * x_t

                c1 = (1 - alpha_hat_t / alpha_hat_next)
                c2 = (1 - alpha_hat_next) / (1 - alpha_hat_t)
                sigma_t = eta * torch.sqrt(c1 * c2)

                z = torch.randn_like(x_t) if time_next > 0 else torch.zeros_like(x_t)
                epsilon_coeff = torch.sqrt(1. - alpha_hat_next - sigma_t**2)
                x_next = torch.sqrt(alpha_hat_next) * pred_x0 + epsilon_coeff * pred_epsilon + sigma_t * z
                x_t = x_next

        return x_t

    def one_epoch(self, is_train):
        loader = self.train_loader if is_train else self.val_loader
        self.model.train(is_train)

        total_losses, smile_losses, ttm_losses = [], [], []
        batch_grad_norms = []

        pbar = tqdm(loader, leave=False, disable="SLURM_JOB_ID" in os.environ)
        for cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, target_s in pbar:
            with torch.set_grad_enabled(is_train):
                cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, target_s = (
                    cond_s.to(self.device), 
                    cond_ema_s_short.to(self.device), 
                    cond_ema_s_long.to(self.device), 
                    cond_sc.to(self.device), 
                    target_s.to(self.device)
                )
                t = torch.randint(1, self.cfg.noise_steps, (target_s.shape[0],), device=self.device)
                noised_target, v_target = self.noise_surfaces(target_s, t)
                model_input = torch.cat([cond_s, cond_ema_s_short, cond_ema_s_long, noised_target], dim=1)
                predicted_v = self.model(model_input, t, cond_sc)
                
                mse_loss = F.mse_loss(predicted_v, v_target)
                
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
                sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
            
                pred_x0 = sqrt_alpha_hat * noised_target - sqrt_one_minus_alpha_hat * predicted_v
                pred_x0_denorm = denormalize_surface_torch(pred_x0, self.norm_stats['mean'], self.norm_stats['std'])

                smile_loss_p, ttm_loss_p = self.penalty_fn(pred_x0_denorm)

                structural_loss_per_sample = (self.lambda_smile * smile_loss_p + self.lambda_ttm * ttm_loss_p)
                snr = self.alpha_hat / (1 - self.alpha_hat + 1e-8)
                snr_weights = snr[t]
                
                total_loss = mse_loss + (snr_weights * structural_loss_per_sample).mean()

            if is_train:
                self.optimizer.zero_grad()
                
                if torch.isinf(total_loss) or torch.isnan(total_loss):
                    print("--- ERROR: Loss is inf or nan! ---")
                    print(f"MSE Loss: {mse_loss.item()}")
                    print(f"Mean SNR Weight: {snr_weights.mean().item()}")
                    print(f"Mean Structural Loss: {structural_loss_per_sample.mean().item()}")
                    import sys; sys.exit() 
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.15)
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                batch_grad_norms.append(total_norm)
                        
                self.optimizer.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)
                self.ema.update_model_average(self.ema_model, self.model)

            pbar.set_description(f"Loss: {total_loss.item():.4f}")
            total_losses.append(total_loss.item())
            ttm_losses.append(ttm_loss_p.mean().item())
            smile_losses.append(smile_loss_p.mean().item())
        avg_loss = np.mean(total_losses)
        avg_comps = {
            "smile": np.mean(smile_losses),
            "ttm": np.mean(ttm_losses)
        }
        
        return avg_loss, avg_comps, batch_grad_norms

    def fit(self):
        train_losses, val_losses = [], []
        all_grad_norms = []
        best_val_loss = float('inf')
        
        for epoch in range(self.cfg.epochs):
            train_loss, train_comps, train_batch_norms = self.one_epoch(is_train=True)
            val_loss, val_comps, _ = self.one_epoch(is_train=False)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            all_grad_norms.extend(train_batch_norms)
            log_msg = (
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                )
            
            logging.info(log_msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.ema_model.state_dict(), os.path.join("models", self.cfg.run_name, "best_loss_model.pt"))
                logging.info(f"Saved new best EMA model with Val Loss: {best_val_loss:.4f}")
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step(val_loss)
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if (epoch + 1) % self.cfg.plot_every == 0:
                cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, target_s = next(iter(self.val_loader))
                val_dates = self.val_loader.dataset.dates
                
                n_plots = 15
                cond_s = cond_s[:n_plots].to(self.device)
                cond_ema_s_short = cond_ema_s_short[:n_plots].to(self.device)
                cond_ema_s_long = cond_ema_s_long[:n_plots].to(self.device)
                cond_sc = cond_sc[:n_plots].to(self.device)
                target_s = target_s[:n_plots]
                dates_to_plot = val_dates[:n_plots]
                
                sampled_surfaces = self.sample(cond_s, cond_ema_s_short, cond_ema_s_long, cond_sc, n_steps=50)
                batch_metrics = calculate_quality_metrics(
                    sampled_surfaces, target_s, self.norm_stats, self.m_grid, self.ttm_grid
                )
                current_mape = batch_metrics['accuracy_mape']
                current_arbitrage = batch_metrics['arbitrage']
                logging.info(f"--- Epoch {epoch+1} Evaluation (on one batch) ---")
                logging.info(f"Batch Val MAPE: {current_mape:.2f}% | Batch Val Arbitrage: {current_arbitrage:.4f}")

                if self.trial:
                     trial_plot_dir = os.path.join("results", self.cfg.run_name, f"trial_{self.trial.number}")
                else:
                    trial_plot_dir = os.path.join("results", self.cfg.run_name)
                
                os.makedirs(trial_plot_dir, exist_ok=True)
                plot_surfaces(
                    surfaces=sampled_surfaces, 
                    dates=dates_to_plot, 
                    title=f"Generated Surfaces Validation Set - Epoch {epoch+1}", 
                    epoch=epoch+1, 
                    save_dir=trial_plot_dir,
                    norm_stats=self.norm_stats
                )
        if self.trial:
            trial_plot_dir = os.path.join("results", self.cfg.run_name, f"trial_{self.trial.number}")
            os.makedirs(trial_plot_dir, exist_ok=True)
            
            plot_loss_curves(train_losses, val_losses, trial_plot_dir)
            plot_gradient_norms(all_grad_norms, trial_plot_dir)
            plot_lr_curve(self.lr_history, trial_plot_dir)
        else:
            base_results_dir = os.path.join("results", self.cfg.run_name)
            plot_loss_curves(train_losses, val_losses, base_results_dir)
            plot_gradient_norms(all_grad_norms, base_results_dir)
            plot_lr_curve(self.lr_history, base_results_dir)
            
        return best_val_loss
import numpy as np
import torch

# Relative imports
from .utils import denormalize_surface
from .financial_utils import penalty_mutau, calculate_penalties_vectorized

# --- Metric Calculation ---

def calculate_quality_metrics(predicted_surfaces_norm, real_surfaces_norm, norm_stats, m_grid, ttm_grid):

    pred_surfaces_denorm = denormalize_surface(predicted_surfaces_norm, norm_stats['mean'], norm_stats['std'])
    real_surfaces_denorm = denormalize_surface(real_surfaces_norm, norm_stats['mean'], norm_stats['std'])
    
    # Calculate arbitrage penalties
    P_T, P_K, PB_K = penalty_mutau(m_grid, ttm_grid)
    _, _, _, penalties = calculate_penalties_vectorized(
        pred_surfaces_denorm, m_grid, ttm_grid, P_T, P_K, PB_K
    )
    mean_arbitrage = np.mean(penalties)
    
    # Calculate accuracy (MAE)
    epsilon = 1e-9 
    mape = 100 * np.mean(np.abs(pred_surfaces_denorm - real_surfaces_denorm) / (np.abs(real_surfaces_denorm) + epsilon))

    return {
        "arbitrage": mean_arbitrage,
        "accuracy_mape": mape, # Return MAPE instead of MAE and min/max
    }

def calculate_mape_per_sample_day(all_samples, true_surfaces):
    """Calculates the MAPE for each individual sample on each day."""
    epsilon = 1e-9
    # Shape becomes: (1, n_days, H, W)
    true_surfaces_expanded = true_surfaces[np.newaxis, :, :, :]
    # Shape of all_samples is (n_samples, n_days, H, W)
    abs_pct_error = 100 * np.abs(all_samples - true_surfaces_expanded) / (np.abs(true_surfaces_expanded) + epsilon)
    # The result has shape (n_samples, n_days)
    mape_per_sample_day = np.mean(abs_pct_error, axis=(2, 3))
    
    return mape_per_sample_day

def calculate_normalized_deviation(true_values, lower_bounds, upper_bounds):
    """
    Calculates the normalized deviation of true values from the center of an interval.
    A value < 1 means the true value is within the [lower, upper] bounds.
    A value > 1 means the true value is outside the bounds.
    """
    epsilon = 1e-9
    
    # Calculate the center and half-width of the confidence interval
    interval_center = (upper_bounds + lower_bounds) / 2
    interval_half_width = (upper_bounds - lower_bounds) / 2
    
    # Calculate the absolute distance of the true value from the center
    distance_from_center = np.abs(true_values - interval_center)
    
    # Normalize the distance by the half-width
    # Add epsilon to avoid division by zero if the interval width is zero
    normalized_score = distance_from_center / (interval_half_width + epsilon)
    
    return normalized_score

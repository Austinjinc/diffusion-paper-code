import torch
import numpy as np
import pandas as pd
import math
import os
import logging
from matplotlib import pyplot as plt
from matplotlib import ticker 

# Relative import
from .utils import denormalize_surface

# --- All Plotting Functions ---

def plot_surfaces(surfaces, dates, title, epoch, save_dir, norm_stats):

    surfaces = denormalize_surface(surfaces, norm_stats['mean'], norm_stats['std'])

    n_images = surfaces.shape[0]
    if n_images == 0:
        return # Do nothing if there are no surfaces to plot

    # Automatically determine the grid size for the subplots
    n_cols = int(math.ceil(math.sqrt(n_images)))
    n_rows = int(math.ceil(n_images / n_cols))

    fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))

    # Define the coordinate grids
    m_grid = np.linspace(0.6, 1.4, 9)
    ttm_grid = np.array([1 / 252, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1])
    X, Y = np.meshgrid(ttm_grid, m_grid)

    for i in range(n_images):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

        # Ensure the data for plotting is 2D
        im_data = surfaces[i]
        if im_data.ndim == 3 and im_data.shape[0] == 1:
             im_data = im_data[0]

        # Plot the 3D surface
        surf = ax.plot_surface(X, Y, im_data, cmap='viridis', edgecolor='none')
        
        # Set the desired viewing angle and flip the moneyness axis
        ax.view_init(elev=30, azim=-135)
        ax.invert_yaxis()

        # Set titles and labels
        date_str = pd.to_datetime(dates[i]).strftime('%Y-%m-%d')
        ax.set_title(f"Sample for {date_str}", pad=20)
        ax.set_xlabel("TTM")
        ax.set_ylabel("Moneyness")
        ax.set_zlabel("Implied Vol")

        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_iv_slice_comparison_with_ci(dates, true_surfaces, mean_predicted_surfaces, 
                                 lower_bound_surfaces, upper_bound_surfaces,
                                 moneyness_idx, ttm_idx, title, run_name):

    true_slice = true_surfaces[:, moneyness_idx, ttm_idx]
    mean_slice = mean_predicted_surfaces[:, moneyness_idx, ttm_idx]
    lower_slice = lower_bound_surfaces[:, moneyness_idx, ttm_idx]
    upper_slice = upper_bound_surfaces[:, moneyness_idx, ttm_idx]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, true_slice, label='Actual IV', color='black', linewidth=1.3, linestyle='--')
    ax.plot(dates, mean_slice, label='Mean Predicted IV', color='C0')

    # Add the shaded confidence interval
    ax.fill_between(dates, lower_slice, upper_slice, color='C0', alpha=0.3, label='90% Confidence Interval')

    ax.set_xlabel('Date')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    filename = title.lower().replace(" ", "_") + "_with_ci.png"
    save_path = os.path.join("results", run_name, filename)
    plt.savefig(save_path, dpi=200)
    logging.info(f"Saved IV slice plot with CI to {save_path}")
    plt.close(fig)

def plot_arbitrage_comparison(dates, true_penalties, predicted_penalties, run_name):
    """
    Creates a scatter plot comparing the arbitrage penalties of true vs. predicted surfaces.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.scatter(dates, true_penalties, s=15, alpha=0.6, label='Actual Surfaces')
    ax.scatter(dates, predicted_penalties, s=15, alpha=0.6, label='Generated Surfaces')

    ax.set_xlabel('Date')
    ax.set_ylabel('Total Arbitrage Penalty')
    ax.set_title('Arbitrage Penalty Comparison: Actual vs. Generated')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    
    # Save the figure to the run's result folder
    save_path = os.path.join("results", run_name, "arbitrage_comparison_plot.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_mean_surface_grid(surfaces_to_plot, dates_to_plot, run_name, n_plots):
    """Plots a grid of the MEAN generated IV surfaces."""
    n_test_samples = len(surfaces_to_plot)
    if n_plots > n_test_samples:
        n_plots = n_test_samples
    if n_plots == 0:
        return

    indices_to_plot = np.linspace(0, n_test_samples - 1, num=n_plots, dtype=int)

    m_grid = np.linspace(0.6, 1.4, 9)
    ttm_grid = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1])
    X, Y = np.meshgrid(ttm_grid, m_grid)

    ncols = min(n_plots, 5)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Mean Generated Surfaces from Test Set ({n_plots} samples)', fontsize=20)

    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, idx in enumerate(indices_to_plot):
        ax = axes[i]
        surface = surfaces_to_plot[idx]
        date = pd.to_datetime(dates_to_plot[idx]).strftime('%Y-%m-%d')

        ax.plot_surface(X, Y, surface, cmap='viridis', edgecolor='none')
        ax.set_title(date, pad=10)
        ax.set_xlabel("TTM", labelpad=10)
        ax.set_ylabel("Moneyness", labelpad=10)
        ax.view_init(elev=30, azim=-135)
        ax.invert_yaxis()

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("results", run_name, "mean_test_set_grid_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved mean test set grid plot to {save_path}")

def plot_mape_timeseries_with_ci(dates, median_mape, lower_bound, upper_bound, run_name):
    """
    Creates a time series plot of the daily MAPE with a 90% confidence interval.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the median MAPE as a line
    ax.plot(dates, median_mape, color='C0', linestyle='--', label='Median Predicted MAPE')

    # Add the shaded confidence interval for the predicted MAPE
    ax.fill_between(dates, lower_bound, upper_bound, color='C0', alpha=0.3, label='90% Confidence Interval (Predicted)')

    ax.set_xlabel('Date')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Daily MAPE with 90% Confidence Interval on the Test Set')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    
    save_path = os.path.join("results", run_name, "mape_timeseries_with_ci.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved MAPE time series with CI to {save_path}")

def plot_arbitrage_timeseries_with_ci(dates, true_penalties, lower_bound, mean, upper_bound, run_name):
    """
    Creates a time series plot of arbitrage penalties with a 90% confidence interval.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the true arbitrage penalty
    ax.plot(dates, true_penalties, color='red', linestyle='--', label='Actual Arbitrage')
    ax.plot(dates, mean, color='C0', linestyle='--', label='Mean Predicted Arbitrage')

    # Add the shaded confidence interval for the predicted penalties
    ax.fill_between(dates, lower_bound, upper_bound, color='C0', alpha=0.3, label='90% Confidence Interval (Predicted)')

    ax.set_xlabel('Date')
    ax.set_ylabel('Arbitrage Penalty')
    ax.set_title('Daily Arbitrage Penalty with 90% Confidence Interval')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    
    save_path = os.path.join("results", run_name, "arbitrage_timeseries_with_ci.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved arbitrage time series with CI to {save_path}")
    
def plot_normalized_deviation(dates, scores, title, run_name):
    """Creates and saves a time series plot of the normalized deviation score."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, scores, marker='o', linestyle='-', markersize=2, label='Deviation Score')
    
    # Add a horizontal line at y=1, the critical threshold
    ax.axhline(y=1, color='r', linestyle='--', label='Confidence Interval Edge')

    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Deviation Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    
    filename = title.lower().replace(" ", "_") + "_deviation.png"
    save_path = os.path.join("results", run_name, filename)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved normalized deviation plot to {save_path}")

def plot_loss_curves(train_losses, val_losses, save_dir):

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, color='tab:blue')
    axes[0].set_title('Training Loss')
    axes[0].set_ylabel('Loss Log-Scale')
    axes[0].set_yscale('log')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs, val_losses, color='tab:orange')
    axes[1].set_title('Validation Loss')
    axes[1].set_ylabel('Loss Log-Scale')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Add a main title for the entire figure
    fig.suptitle('Training & Validation Loss Over Epochs', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_gradient_norms(grad_norms, save_dir):
    
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(grad_norms)
    
    # Use a log scale on the y-axis to make explosions more visible
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    # 3. Use a standard number format for the labels
    ax.yaxis.set_major_formatter(ticker.LogFormatter(base=10.0))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(True, which='major', linestyle='--', linewidth=1.0) 
    
    ax.set_xlabel('Training Step (Batch Number)')
    ax.set_ylabel('Gradient Norm (Log Scale)')
    ax.set_title('Gradient Norm Over Training')

    fig.tight_layout()
    
    save_path = os.path.join(save_dir, "gradient_norms.png")
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved gradient norm plot to {save_path}")
    plt.close(fig)
    
def plot_lr_curve(lr_history, save_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lr_history)
    ax.set_xlabel('Training Step (Batch Number)')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule Over Training')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "lr_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
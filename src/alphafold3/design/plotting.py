"""Plotting utilities for AlphaFold 3 design trajectories."""

import os
import logging
from typing import Dict, Any, Sequence, Optional
import numpy as np

# Import matplotlib, handle import error gracefully
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("matplotlib not found. Plotting functionality will be disabled.")
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Mapping from trajectory keys to plot labels/filenames
METRIC_MAP = {
    'loss': 'Total Loss',
    'plddt_score': 'pLDDT Score', # Updated key from previous steps
    'pae_loss': 'PAE Loss (Normalized)', # Updated key from previous steps
    'contact_inter': 'Contact Entropy (Inter)',
    'contact_intra': 'Contact Entropy (Intra)',
    # Add other metrics if they exist in the trajectory dict
}

def _plot_metric(
    steps: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str,
    output_dir: str,
    filename_prefix: str,
    ax: Optional[plt.Axes] = None # Allow passing an Axes object
) -> None:
    """Helper to plot a single metric over steps."""
    if len(steps) == 0 or len(metric_values) == 0:
        logger.debug(f"Skipping plot for '{metric_name}': No data points.")
        return
    if len(steps) != len(metric_values):
        logger.debug(f"Skipping plot for '{metric_name}': Data length mismatch (steps={len(steps)}, metric={len(metric_values)}).")
        return

    standalone_plot = ax is None
    if standalone_plot:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(steps, metric_values, marker='.', linestyle='-')
    ax.set_title(f"Design Trajectory: {metric_name}")
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel(metric_name)
    ax.grid(True, linestyle='--', alpha=0.6)

    if standalone_plot:
        plot_filename = os.path.join(output_dir, f"{filename_prefix}_{metric_name.lower().replace(' ', '_')}.png")
        try:
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=150)
            logger.debug(f"Saved plot: {plot_filename}")
            plt.close(fig) # Close the figure to free memory
        except Exception as e:
            logger.error(f"Failed to save plot {plot_filename}: {e}")

def plot_design_trajectory_summary(
    design_results: Dict[str, Any],
    output_dir: str,
    plot_filename_prefix: str,
) -> None:
    """Generates a single multi-panel plot summarizing the design trajectory."""
    if "trajectory" not in design_results:
        logger.warning("Trajectory data not found in design_results. Skipping summary plot.")
        return

    traj = design_results["trajectory"]
    steps = np.array(traj.get("step", []))
    if len(steps) == 0:
        logger.warning("No steps found in trajectory data. Skipping summary plot.")
        return

    # Define metrics to plot and their corresponding keys in the trajectory dict
    metrics_to_plot = {
        "Total Loss": "loss",
        "pLDDT Score": "plddt", # Assuming plddt is stored as score [0,1]
        "PAE Loss": "pae_loss",
        "Contact Inter Loss": "contact_inter",
        "Contact Intra Loss": "contact_intra",
    }

    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 3 * num_metrics), sharex=True)
    fig.suptitle(f"{plot_filename_prefix} Design Trajectory Summary", fontsize=14)

    for i, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
        metric_values = np.array(traj.get(metric_key, []))
        ax = axes[i] if num_metrics > 1 else axes # Handle single metric case

        if len(metric_values) == len(steps):
            _plot_metric(steps, metric_values, metric_name, output_dir, plot_filename_prefix, ax=ax)
        else:
            logger.warning(f"Skipping subplot for '{metric_name}': Data length mismatch or missing key '{metric_key}'.")
            ax.set_title(f"{metric_name} (Data Unavailable)")
            ax.text(0.5, 0.5, 'Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.grid(True, linestyle='--', alpha=0.6)


    axes[-1].set_xlabel("Iteration Step") # Set xlabel only on the last subplot

    summary_plot_filename = os.path.join(output_dir, f"{plot_filename_prefix}_summary.png")
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
        plt.savefig(summary_plot_filename, dpi=150)
        logger.info(f"Saved summary plot: {summary_plot_filename}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to save summary plot {summary_plot_filename}: {e}")

def plot_design_trajectory(
    design_results: Dict[str, Any],
    output_dir: str,
    plot_filename_prefix: str,
) -> None:
    """Generates plots visualizing the design trajectory metrics."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Generating trajectory plots in: {plots_dir}")

    if "trajectory" not in design_results:
        logger.warning("Trajectory data not found in design_results. Skipping plots.")
        return

    # --- Call the new summary plot function ---
    plot_design_trajectory_summary(design_results, plots_dir, plot_filename_prefix)
    # ------------------------------------------

    # --- Keep individual plots for now (optional) ---
    # traj = design_results["trajectory"]
    # steps = np.array(traj.get("step", []))
    #
    # metrics_to_plot = {
    #     "Total Loss": "loss",
    #     "pLDDT Score": "plddt",
    #     "PAE Loss": "pae_loss",
    #     "Contact Inter Loss": "contact_inter",
    #     "Contact Intra Loss": "contact_intra",
    # }
    #
    # for metric_name, metric_key in metrics_to_plot.items():
    #     metric_values = np.array(traj.get(metric_key, []))
    #     _plot_metric(steps, metric_values, metric_name, plots_dir, plot_filename_prefix)
    # --------------------------------------------

    logger.info("Finished generating trajectory plots.") 

# Placeholder for future animation/3D plot functions
# def plot_trajectory_animation(...) etc. 
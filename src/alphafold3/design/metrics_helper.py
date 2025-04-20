"""Helper functions to integrate metrics collection with binder design."""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from alphafold3.design import metrics
from alphafold3.design import logging as af_logging

# Get module logger
logger = af_logging.get_logger(__name__)


def create_metrics_collector_for_design(
    output_dir: str,
    run_name: str,
    design_params: Dict[str, Any],
    format: str = "json",
    write_interval: int = 10,
) -> metrics.MetricsCollector:
    """Create a metrics collector for a design run.
    
    Args:
        output_dir: Directory to write metrics files to
        run_name: Name of this run (used in filenames)
        design_params: Design parameters from run_alphafold.py
        format: Output format ('json' or 'csv')
        write_interval: How often to write metrics (every N steps)
        
    Returns:
        Initialized MetricsCollector
    """
    # Use the protocol name in the metrics filename
    protocol = design_params.get("protocol", "unknown")
    metrics_name = f"{run_name}_{protocol}"
    
    return metrics.create_metrics_collector(
        output_dir=output_dir,
        run_name=metrics_name,
        format=format,
        write_interval=write_interval,
    )


def log_design_step(
    collector: metrics.MetricsCollector,
    step: int,
    loss: float,
    loss_breakdown: Dict[str, Any],
    sequence: Optional[str] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Log metrics for a design step.
    
    Args:
        collector: MetricsCollector instance
        step: Current step number
        loss: Overall loss value
        loss_breakdown: Dictionary of individual loss components
        sequence: Optional current sequence
        additional_metrics: Optional additional metrics
    """
    # Combine all metrics
    step_metrics = {
        "loss": loss,  # Total loss
    }
    
    # Add individual loss components
    for k, v in loss_breakdown.items():
        # Prefix with loss_ to avoid name clashes
        step_metrics[f"loss_{k}"] = v
    
    # Add sequence if provided
    if sequence:
        step_metrics["sequence"] = sequence
    
    # Add any additional metrics
    if additional_metrics:
        step_metrics.update(additional_metrics)
    
    # Record the metrics
    collector.record(step=step, metrics=step_metrics)


def format_design_summary(
    design_results: Dict[str, Any],
    metrics_collector: Optional[metrics.MetricsCollector] = None,
) -> Dict[str, Any]:
    """Create a clean summary of design results for serialization.
    
    Args:
        design_results: Raw design results dictionary
        metrics_collector: Optional metrics collector with the full history
        
    Returns:
        Dictionary with clean, serializable results
    """
    # Start with a clean slate
    summary = {
        "protocol": design_results.get("protocol", "unknown"),
        "design_time": design_results.get("design_time", 0.0),
        "success": design_results.get("success", True),
    }
    
    # Add best metrics from the collector if available
    if metrics_collector:
        # Get best loss
        best_loss_metrics = metrics_collector.get_best_metrics(key="loss", mode="min")
        if best_loss_metrics:
            summary["best_metrics"] = {
                "step": best_loss_metrics.get("step", 0),
                "loss": best_loss_metrics.get("loss", 0.0),
            }
            
            # Add other available metrics from the best step
            for k, v in best_loss_metrics.items():
                if k.startswith("loss_") and isinstance(v, (int, float)):
                    # Extract loss component name
                    component = k[5:]  # Remove "loss_" prefix
                    summary["best_metrics"][component] = v
            
            # Add pLDDT if available
            if "loss_plddt" in best_loss_metrics:
                # pLDDT is negative in the loss, convert back
                summary["best_metrics"]["plddt"] = -best_loss_metrics["loss_plddt"]
    
    # Handle the designed sequence
    if "final_designed_sequence" in design_results:
        summary["designed_sequence"] = design_results["final_designed_sequence"]
    elif "final_designed_sequences" in design_results:
        summary["designed_sequences"] = design_results["final_designed_sequences"]
    
    # Extract final AA indices if available
    if "final_aa_indices" in design_results:
        summary["final_aa_indices"] = design_results["final_aa_indices"]
    
    return summary


def save_design_results(
    design_results: Dict[str, Any],
    output_dir: str,
    run_name: str,
    seed: int,
    metrics_collector: Optional[metrics.MetricsCollector] = None,
) -> str:
    """Save design results to a JSON file.
    
    Args:
        design_results: Design results from the design process
        output_dir: Directory to write results to
        run_name: Name for this run
        seed: Random seed used
        metrics_collector: Optional metrics collector
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the summary
    summary = format_design_summary(design_results, metrics_collector)
    
    # Include trajectory info from metrics collector
    if metrics_collector:
        # Get all metrics
        all_metrics = metrics_collector.metrics_history
        
        # Process into steps for serialization
        steps = []
        for step_metrics in all_metrics:
            step_data = {
                "step": step_metrics.get("step", 0),
                "loss": step_metrics.get("loss", 0.0),
            }
            
            # Add loss components
            for k, v in step_metrics.items():
                if k.startswith("loss_") and isinstance(v, (int, float)):
                    component = k[5:]  # Remove "loss_" prefix
                    step_data[component] = v
            
            # Add sequence if available
            if "sequence" in step_metrics:
                step_data["sequence"] = step_metrics["sequence"]
                
            steps.append(step_data)
            
        # Add to summary
        summary["trajectory"] = {
            "steps": steps,
            "num_steps": len(steps)
        }
    
    # Path for the results file
    output_path = os.path.join(output_dir, f"{run_name}_seed_{seed}_design_results.json")
    
    # Save the summary
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved design results to {output_path}")
    
    return output_path 
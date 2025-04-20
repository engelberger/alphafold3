"""Metrics collection and persistence utilities for AlphaFold 3 design."""

import os
import json
import csv
import time
import datetime
from typing import Dict, Any, List, Optional, TextIO, Union
import threading

import numpy as np
import jax
import jax.numpy as jnp

# Local imports
from alphafold3.design import logging as af_logging

# Get module logger
logger = af_logging.get_logger(__name__)


class MetricsCollector:
    """Collects and persists metrics during design runs."""
    
    def __init__(
        self,
        output_dir: str,
        run_name: str,
        write_interval: int = 10,
        format: str = "json",
    ):
        """Initialize metrics collector.
        
        Args:
            output_dir: Directory to write metrics files to
            run_name: Name of this run (used in filenames)
            write_interval: How often to write metrics (every N steps)
            format: Output format ('json' or 'csv')
        """
        self.output_dir = output_dir
        self.run_name = run_name
        self.write_interval = write_interval
        self.format = format.lower()
        
        # Create metrics directory
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Prepare output file paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(
            self.metrics_dir, f"{run_name}_{timestamp}.{self.format}"
        )
        
        # Initialize metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()  # For thread-safety
        self._file_handle: Optional[TextIO] = None
        self._csv_writer = None
        self._header_written = False
        
        logger.info(f"MetricsCollector initialized. Output: {self.metrics_file}")
    
    def __del__(self):
        """Ensure file is closed on garbage collection."""
        self.close()
    
    def close(self):
        """Close file handles."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def _prepare_file(self, metrics: Dict[str, Any]):
        """Prepare output file based on format."""
        if self._file_handle is not None:
            return
            
        try:
            if self.format == 'csv':
                self._file_handle = open(self.metrics_file, 'w', newline='')
                self._csv_writer = csv.DictWriter(
                    self._file_handle, 
                    fieldnames=list(metrics.keys())
                )
            else:  # json/jsonl
                self._file_handle = open(self.metrics_file, 'w')
        except IOError as e:
            logger.error(f"Error creating metrics file {self.metrics_file}: {e}")
            self._file_handle = None
    
    def _write_metrics(self, metrics_list: List[Dict[str, Any]]):
        """Write metrics to file."""
        if not metrics_list:
            return
            
        try:
            self._prepare_file(metrics_list[0])
            
            if not self._file_handle:
                return
                
            if self.format == 'csv':
                if not self._header_written:
                    self._csv_writer.writeheader()
                    self._header_written = True
                
                for metrics in metrics_list:
                    self._csv_writer.writerow(metrics)
                self._file_handle.flush()
            else:  # json/jsonl format
                for metrics in metrics_list:
                    self._file_handle.write(json.dumps(metrics) + "\n")
                self._file_handle.flush()
                
        except (IOError, ValueError) as e:
            logger.error(f"Error writing metrics: {e}")
    
    def record(self, step: int, metrics: Dict[str, Any]):
        """Record a new metrics point.
        
        Args:
            step: Current optimization step
            metrics: Dictionary of metric values
        """
        # Convert JAX arrays to Python/NumPy values for serialization
        processed_metrics = {}
        
        for k, v in metrics.items():
            if isinstance(v, (jnp.ndarray, jax.Array)):
                try:
                    processed_metrics[k] = float(v)
                except (TypeError, ValueError):
                    # Skip complex arrays
                    continue
            elif isinstance(v, np.ndarray):
                if v.size == 1:
                    processed_metrics[k] = float(v.item())
                else:
                    # Skip complex arrays
                    continue
            elif isinstance(v, (int, float, bool, str)):
                processed_metrics[k] = v
            # Skip other types
        
        # Add metadata
        processed_metrics['step'] = step
        processed_metrics['timestamp'] = time.time()
        
        # Add to history
        with self._lock:
            self.metrics_history.append(processed_metrics)
            
            # Periodically write to file
            if step % self.write_interval == 0 or step == 0:
                # Copy list to avoid issues if metrics are added during writing
                metrics_to_write = list(self.metrics_history)
                self._write_metrics(metrics_to_write)
                
        # Log metrics at appropriate intervals
        if step % 10 == 0 or step == 0:
            af_logging.log_with_metrics(
                logger, 
                level=af_logging.logging.INFO,
                msg=f"Design step {step} metrics",
                metrics=processed_metrics
            )
    
    def get_best_metrics(self, key: str, mode: str = 'min') -> Dict[str, Any]:
        """Get metrics entry with best value for given key.
        
        Args:
            key: Metric name to optimize
            mode: 'min' or 'max' to minimize or maximize
            
        Returns:
            Dictionary with best metrics
        """
        if not self.metrics_history:
            return {}
            
        with self._lock:
            valid_metrics = [m for m in self.metrics_history if key in m]
            
            if not valid_metrics:
                return {}
                
            if mode.lower() == 'min':
                best_idx = np.argmin([m[key] for m in valid_metrics])
            else:
                best_idx = np.argmax([m[key] for m in valid_metrics])
                
            return valid_metrics[best_idx]
    
    def get_last_metrics(self) -> Dict[str, Any]:
        """Get most recent metrics entry."""
        with self._lock:
            if not self.metrics_history:
                return {}
            return self.metrics_history[-1]


def create_metrics_collector(
    output_dir: str,
    run_name: str,
    **kwargs
) -> MetricsCollector:
    """Factory function to create a MetricsCollector.
    
    Args:
        output_dir: Directory to write metrics files to
        run_name: Name of this run (used in filenames)
        **kwargs: Additional arguments passed to MetricsCollector
        
    Returns:
        Initialized MetricsCollector
    """
    return MetricsCollector(output_dir, run_name, **kwargs)


def convert_jax_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JAX values in metrics to Python types.
    
    Args:
        metrics: Dictionary containing metrics which may include JAX values
        
    Returns:
        Dictionary with JAX values converted to Python types
    """
    result = {}
    
    for k, v in metrics.items():
        if isinstance(v, (jnp.ndarray, jax.Array)):
            try:
                # Convert to Python scalar if possible
                result[k] = float(v)
            except (TypeError, ValueError):
                # Skip complex arrays
                continue
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                result[k] = float(v.item())
            else:
                # Skip complex arrays 
                continue
        elif isinstance(v, (int, float, bool, str, type(None))):
            result[k] = v
        # Skip other types
    
    return result 
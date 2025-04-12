"""Utilities for memory management in AlphaFold 3.

This module provides utilities for explicitly managing memory during
computationally intensive operations like gradient-based design.
"""

import gc
import logging
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    logging.warning("JAX not available, some memory utilities will be unavailable")

def clear_gpu_memory(force: bool = False) -> None:
    """Clear JAX GPU memory cache.
    
    Args:
        force: Whether to force clearing even on non-GPU devices.
    """
    try:
        # Check if we're running on GPU
        if not force and not any('gpu' in d.lower() for d in jax.devices()):
            logging.debug("Not running on GPU, skipping GPU memory clear")
            return
            
        # Clear JAX GPU memory
        logging.info("Clearing JAX GPU memory cache")
        backend = jax.lib.xla_bridge.get_backend()
        if hasattr(backend, 'clear_cache'):
            backend.clear_cache()
        logging.info("JAX GPU memory cache cleared")
    except Exception as e:
        logging.warning(f"Error clearing JAX GPU memory: {e}")

def clear_cpu_memory() -> None:
    """Force Python garbage collection to free CPU memory."""
    logging.info("Running Python garbage collection")
    gc.collect()
    logging.info("Python garbage collection completed")

def clear_memory(include_gpu: bool = True, include_cpu: bool = True) -> None:
    """Clear both GPU and CPU memory caches.
    
    Args:
        include_gpu: Whether to clear GPU memory.
        include_cpu: Whether to clear CPU memory.
    """
    if include_gpu:
        clear_gpu_memory()
    if include_cpu:
        clear_cpu_memory()
    logging.info("Memory clearing completed")

def get_current_memory_usage() -> Optional[str]:
    """Get the current memory usage information.
    
    Returns:
        String with memory usage info or None if not available.
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Try to get GPU memory info if running on GPU
        gpu_mem_info = ""
        if any('gpu' in d.lower() for d in jax.devices()):
            try:
                from jax.experimental.maps import xmap
                from jax.experimental.pjit import pjit
                
                # This is a hacky way to get GPU memory info, might not work on all setups
                gpu_mem_info = f", GPU mem: {jax.devices()[0].memory_stats()}"
            except:
                gpu_mem_info = " (GPU mem info unavailable)"
        
        return f"Memory usage: RSS={mem_info.rss/(1024**3):.2f}GB, VMS={mem_info.vms/(1024**3):.2f}GB{gpu_mem_info}"
    except Exception as e:
        logging.warning(f"Could not get memory usage: {e}")
        return None 
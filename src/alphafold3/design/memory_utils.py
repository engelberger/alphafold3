"""Utilities for managing memory during computationally intensive operations."""

import gc
import jax

def clear_mem():
    """Clear GPU memory buffers and trigger garbage collection.
    
    This function attempts to free memory in two ways:
    1. Deleting JAX live buffers on the GPU/TPU to free device memory
    2. Running Python's garbage collector to free CPU memory
    
    Note that clearing JAX buffers may invalidate previously computed values
    that haven't been explicitly saved.
    """
    # Clear VRAM (GPU/TPU memory)
    backend = jax.lib.xla_bridge.get_backend()
    if hasattr(backend, 'live_buffers'):
        for buf in backend.live_buffers():
            buf.delete()
    
    # Clear RAM (CPU memory)
    gc.collect() 
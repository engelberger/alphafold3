"""Utility functions for binder design protocols."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Optional

def freeze_containers_for_jax(obj):
    """Makes a nested structure of dicts and lists JAX-compatible by making them immutable.

    Args:
        obj: A nested structure of dicts, lists and leaf values.

    Returns:
        A similar structure with dicts and lists converted to immutable types.
    """
    if isinstance(obj, dict):
        # Convert each value in the dict and return an immutable mapping (FrozenDict)
        return jax.tree_util.tree_map(
            freeze_containers_for_jax,
            {k: v for k, v in obj.items() if not (isinstance(v, np.ndarray) and v.dtype == object)}
        )
    elif isinstance(obj, list):
        # Convert each element in the list and return a tuple (immutable)
        return tuple(freeze_containers_for_jax(x) for x in obj)
    else:
        # For leaf values (including JAX arrays), return as is
        return obj

def safe_jax_to_float(x):
    """Safely convert JAX array to float for logging, handling potential errors."""
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float('nan')

def safe_process_losses(losses):
    """Process dictionary of JAX losses to Python values for logging."""
    if not isinstance(losses, dict):
        return {"loss": safe_jax_to_float(losses)}

    return {k: safe_jax_to_float(v) if hasattr(v, "item") or not isinstance(v, dict)
            else safe_process_losses(v) for k, v in losses.items()}

def print_log_line(prefix: str, log_data: Dict[str, Any], keys: Optional[List[str]] = None) -> None:
    """Prints a formatted log line with selected keys from log_data."""
    if keys is None:
        keys = sorted(log_data.keys())
    items: List[str] = []
    for key in keys:
        if key in log_data:
            val = log_data[key]
            if isinstance(val, float):
                items.append(f"{key}={val:.3f}")
            else:
                items.append(f"{key}={val}")
    line = prefix + " | " + ", ".join(items)
    print(line) 
"""Loss functions related to sequence properties, like entropy."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple

from alphafold3.design.losses.common import safe_mean

# Moved from binder_loss.py
def get_binder_seq_entropy_loss(binder_seq_logits):
    """Calculate sequence entropy loss to encourage diversity.

    Args:
        binder_seq_logits: Sequence logits for the binder (JAX array).

    Returns:
        Negative mean sequence entropy (scalar JAX array).
    """
    # Ensure input is JAX array
    binder_seq_logits = jnp.asarray(binder_seq_logits)

    probs = jax.nn.softmax(binder_seq_logits, axis=-1)
    # Add epsilon for numerical stability before log
    log_probs = jnp.log(jnp.maximum(probs, 1e-8)) # Use maximum instead of add
    entropy = -jnp.sum(probs * log_probs, axis=-1) # Entropy per position

    # Return negative mean entropy (as we want to maximize entropy for diversity)
    # Use nan_to_num for safety before final mean
    return -jnp.mean(jnp.nan_to_num(entropy))

# Placeholder - Functions will be moved here 
"""Loss functions related to sequence properties, like entropy."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
import logging

from alphafold3.design.losses.common import safe_mean

logger = logging.getLogger(__name__)

# Moved from binder_loss.py
def get_binder_seq_entropy_loss(binder_seq_logits):
    """Calculate sequence entropy loss to encourage diversity.

    Args:
        binder_seq_logits: Sequence logits for the binder (JAX array).

    Returns:
        Negative mean sequence entropy (scalar JAX array).
    """
    logger.debug(
        f"get_binder_seq_entropy_loss called with binder_seq_logits.shape={getattr(binder_seq_logits, 'shape', None)}"
    )
    # Ensure input is JAX array
    binder_seq_logits = jnp.asarray(binder_seq_logits)
    logger.debug(f"binder_seq_logits after asarray shape={binder_seq_logits.shape}")

    probs = jax.nn.softmax(binder_seq_logits, axis=-1)
    logger.debug(f"probs first 5 entries={probs.flatten()[:5]}")
    # Add epsilon for numerical stability before log
    log_probs = jnp.log(jnp.maximum(probs, 1e-8)) # Use maximum instead of add
    entropy = -jnp.sum(probs * log_probs, axis=-1) # Entropy per position
    logger.debug(f"entropy per position sample={entropy[:5]}")

    # Return negative mean entropy (as we want to maximize entropy for diversity)
    # Use nan_to_num for safety before final mean
    loss = -jnp.mean(jnp.nan_to_num(entropy))
    logger.debug(f"get_binder_seq_entropy_loss returning loss={loss}")
    return loss

# Placeholder - Functions will be moved here 
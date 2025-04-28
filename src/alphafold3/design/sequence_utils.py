"""Utilities for sequence manipulation in design protocols."""

import jax
import jax.numpy as jnp
from typing import Dict, Any

# Define alphabet size for clarity, assuming standard 20 AAs
ALPHABET_SIZE = 20

def soft_seq_af3(
    logits: jnp.ndarray,
    bias: jnp.ndarray,
    opt: Dict[str, Any],
    key: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Generates different sequence representations using Straight-Through Estimator (STE).

    Analogous to ColabDesign's `shared.model.soft_seq`.

    Args:
        logits: Sequence logits (shape [..., L, ALPHABET_SIZE]).
        bias: Sequence bias (shape [L, ALPHABET_SIZE]).
        opt: Dictionary containing optimization options:
            'alpha': Scaling factor for logits (float).
            'temp': Temperature for softmax (float).
            'soft': Weight for soft representation in pseudo sequence (float, 0-1).
            'hard': Weight for STE representation in pseudo sequence (float, 0-1).
        key: JAX random key (unused in current STE implementation, but kept for
             potential future use like Gumbel softmax).

    Returns:
        A dictionary containing:
            'logits': Scaled logits (logits * alpha + bias).
            'pssm': Position-specific scoring matrix (softmax of scaled_logits).
            'soft': Softmax representation with temperature.
            'hard': One-hot representation based on argmax of 'soft' (with STE).
            'pseudo': Mixed representation combining soft and hard based on opt flags.
    """
    # Ensure opt values exist with defaults if necessary
    alpha = opt.get('alpha', 1.0)
    temp = opt.get('temp', 1.0)
    soft_weight = opt.get('soft', 0.0) # Default to no soft component if not specified
    hard_weight = opt.get('hard', 0.0) # Default to no hard component if not specified

    # Apply bias and alpha scaling
    # Ensure bias can be broadcasted if logits have batch dimension
    if bias.ndim < logits.ndim:
        bias_expanded = bias.reshape((1,) * (logits.ndim - bias.ndim) + bias.shape)
    else:
        bias_expanded = bias
    scaled_logits = logits * alpha + bias_expanded

    # Calculate PSSM (standard softmax)
    pssm = jax.nn.softmax(scaled_logits, axis=-1)

    # Calculate soft representation (softmax with temperature)
    soft = jax.nn.softmax(scaled_logits / temp, axis=-1)

    # Calculate hard representation (one-hot with STE)
    hard_indices = jnp.argmax(soft, axis=-1)
    hard_one_hot = jax.nn.one_hot(hard_indices, num_classes=ALPHABET_SIZE)
    # Straight-Through Estimator (STE)
    ste_hard = jax.lax.stop_gradient(hard_one_hot - soft) + soft

    # Calculate pseudo sequence (mixture based on weights)
    # Start with the base PSSM (softmax of original logits, equivalent to soft_weight=0, hard_weight=0)
    # Note: ColabDesign uses seq['input'], here using softmax(logits) or pssm is analogous
    # if logits are the primary trainable parameters. Using pssm directly.
    pseudo_base = pssm # Represents the non-ste/non-soft component

    # Mix in soft component
    pseudo = (1.0 - soft_weight) * pseudo_base + soft_weight * soft

    # Mix in hard (STE) component
    pseudo = (1.0 - hard_weight) * pseudo + hard_weight * ste_hard

    # Return dictionary of representations
    return {
        'logits': scaled_logits,
        'pssm': pssm,
        'soft': soft,
        'hard': ste_hard,
        'pseudo': pseudo
    } 
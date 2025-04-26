"""Common utilities for loss functions.

This module contains shared utilities used across different loss modules,
such as functions for handling masks, safe aggregation, and other
common operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
import functools

from alphafold3.design import logging

# Get module logger
logger = logging.get_logger(__name__)

# Define constants for cutoffs (adjust if needed)
INTRA_CUTOFF_ANGSTROM = 14.0
INTER_CUTOFF_ANGSTROM = 22.0

def safe_mean(x, axis=None, keepdims=False):
    """Calculates mean avoiding NaNs."""
    return jnp.nansum(x, axis=axis, keepdims=keepdims) / jnp.maximum(
        1.0, jnp.sum(~jnp.isnan(x), axis=axis, keepdims=keepdims)
    )

# Note: Assumes distogram_logits has shape (..., N, N, num_bins)
# and bin_edges has shape (num_bins - 1,)
def entropy_low_bins(distogram_logits, bin_edges, cutoff_angstrom):
    """Calculates entropy over distogram bins below a distance cutoff.

    Matches Equations 2-5 in Bohnuud et al., BioRxiv 2024 (BoltzDesign1).

    Args:
        distogram_logits: Raw logits output from the distogram head (*, N, N, B).
        bin_edges: Edges defining the B-1 bins (B-1,).
        cutoff_angstrom: The distance threshold (e.g., 14.0 or 22.0).

    Returns:
        Entropy calculated over bins below the cutoff (per pair) (*, N, N).
    """
    logger.debug(
        f"entropy_low_bins called with cutoff={cutoff_angstrom}, "
        f"distogram_logits.shape={getattr(distogram_logits, 'shape', None)}, "
        f"bin_edges.shape={getattr(bin_edges, 'shape', None)}"
    )
    if distogram_logits is None or bin_edges is None:
        # Return a placeholder or raise an error if inputs are missing
        # Returning zeros might silently hide issues
        raise ValueError("Distogram logits or bin edges are None in entropy_low_bins")
        # return jnp.zeros(distogram_logits.shape[:-1])

    # Calculate bin centers or upper bounds to compare with cutoff
    # Using upper bounds matches the paper's intent (bins_high <= cutoff)
    bin_width = bin_edges[1] - bin_edges[0] # Assume uniform bins
    bins_high = jnp.append(bin_edges, bin_edges[-1] + bin_width)
    logger.debug(f"bins_high: {bins_high}")

    # Create mask for bins below the cutoff
    low_bin_mask = (bins_high <= cutoff_angstrom).astype(jnp.float32)
    logger.debug(f"low_bin_mask sum: {jnp.sum(low_bin_mask)}, mask shape: {low_bin_mask.shape}")

    # Ensure mask has correct dimensions for broadcasting: (1, 1, B)
    low_bin_mask = low_bin_mask.reshape((1,) * (distogram_logits.ndim - 1) + (-1,))

    # Calculate softmax over all bins (q)
    q = jax.nn.softmax(distogram_logits, axis=-1) # Shape (*, N, N, B)
    logger.debug(f"q first 5 entries: {q.flatten()[:5]}")

    # Calculate masked softmax (q_star) using logit trick for numerical stability
    q_star_logits = distogram_logits - 1e7 * (1.0 - low_bin_mask)
    q_star = jax.nn.softmax(q_star_logits, axis=-1) # Eq 5: Softmax over masked logits
    logger.debug(f"q_star first 5 entries: {q_star.flatten()[:5]}")

    # Calculate entropy using q_star to weight the log probabilities of q (Eq 2)
    # Add epsilon for log stability
    entropy = -jnp.sum(q_star * jnp.log(q + 1e-9), axis=-1) # Shape (*, N, N)
    logger.debug(f"entropy shape: {entropy.shape}, first 5 entries: {entropy.flatten()[:5]}")

    return entropy

# Pre-compute partial functions for common cutoffs if bin_edges are static
# This requires knowing the bin_edges at import time, which might not be true.
# Instead, create closures inside the main loss function where bin_edges are available.
# Example placeholder:
# def get_intra_entropy_fn(bin_edges):
#     return functools.partial(entropy_low_bins, bin_edges=bin_edges, cutoff_angstrom=INTRA_CUTOFF_ANGSTROM)
#
# def get_inter_entropy_fn(bin_edges):
#     return functools.partial(entropy_low_bins, bin_edges=bin_edges, cutoff_angstrom=INTER_CUTOFF_ANGSTROM)


def safe_jax_to_float(x: Any) -> float:
    """Safely convert JAX array to float for logging, handling potential errors.
    
    Args:
        x: Value to convert (JAX array, numpy array, or scalar)
        
    Returns:
        Float value or NaN if conversion fails
    """
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float('nan')


def safe_process_losses(losses: Dict[str, Any]) -> Dict[str, Any]:
    """Process dictionary of JAX losses to Python values for logging.
    
    Args:
        losses: Dictionary containing losses which may include JAX values
        
    Returns:
        Dictionary with JAX values converted to Python types
    """
    if not isinstance(losses, dict):
        return {"loss": safe_jax_to_float(losses)}
    
    return {k: safe_jax_to_float(v) if hasattr(v, "item") or not isinstance(v, dict) 
            else safe_process_losses(v) for k, v in losses.items()}


def get_residue_mask(
    num_residues: int,
    indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Creates a binary mask for the specified residue indices.
    
    Args:
        num_residues: Total number of residues in the protein
        indices: Indices to include in the mask, if None all residues are included
        
    Returns:
        Binary mask of shape [num_residues] with 1.0 for included residues
    """
    if indices is None:
        return jnp.ones(num_residues)
    
    mask = jnp.zeros(num_residues)
    mask = mask.at[indices].set(1.0)
    return mask


def mask_interfaces(matrix, row_indices, col_indices):
    """Create a masked version of a matrix focused on specified interfaces.

    Args:
        matrix: Input matrix to mask.
        row_indices: Indices along the row dimension.
        col_indices: Indices along the column dimension.

    Returns:
        Masked matrix containing only the values at the intersection of row_indices and col_indices.
    """
    # Ensure indices are JAX arrays
    row_indices = jnp.asarray(row_indices)
    col_indices = jnp.asarray(col_indices)
    
    # Create a mask of zeros
    mask = jnp.zeros_like(matrix)
    
    # Set 1s at the interface positions
    def create_interface_mask(m):
        return m.at[jnp.ix_(row_indices, col_indices)].set(1.0)
    
    # Only apply masking if matrix is 2D (otherwise return zeros)
    mask = jax.lax.cond(
        jnp.equal(matrix.ndim, 2),
        create_interface_mask,
        lambda m: m,
        mask
    )
    
    # Apply the mask
    return matrix * mask


def mask_by_threshold(matrix, reference, threshold, direction='greater'):
    """Apply a threshold mask to a matrix based on values in a reference matrix.
    
    Args:
        matrix: Matrix to mask.
        reference: Reference matrix with values to compare against the threshold.
        threshold: Threshold value for masking.
        direction: Either 'greater' for keeping values where reference > threshold,
                  or 'less' for keeping values where reference < threshold.
    
    Returns:
        Masked matrix where elements are set to NaN based on the threshold condition.
    """
    if direction == 'greater':
        condition = reference > threshold
    elif direction == 'less':
        condition = reference < threshold
    else:
        raise ValueError(f"Direction must be 'greater' or 'less', got {direction}")
    
    # Where condition is False, set to NaN, otherwise keep original value
    return jnp.where(condition, matrix, jnp.nan)


def get_matrix_from_result(result, key_priority=None):
    """Extract a matrix from a model result dictionary based on key priority.
    
    Args:
        result: Model result dictionary.
        key_priority: List of keys to try in order of preference.
        
    Returns:
        Extracted matrix as a JAX array, or None if not found.
    """
    if key_priority is None:
        return None
        
    if not isinstance(result, dict):
        return None
    
    # Try each key in order of priority
    for key in key_priority:
        if key in result:
            return jnp.asarray(result[key])
    
    return None


def normalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
    """Normalize weights to sum to 1.0, safely handling zero sums.
    
    Args:
        weights: Input array of weights
        
    Returns:
        Normalized weights that sum to 1.0
    """
    sum_weights = jnp.sum(weights)
    return jnp.where(sum_weights > 0.0, weights / sum_weights, weights)


def softmax_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Calculate softmax cross entropy loss.
    
    Args:
        logits: Unnormalized log probabilities
        targets: Target probabilities or one-hot encoded labels
        weights: Optional weights for each example
        
    Returns:
        Weighted softmax cross entropy loss
    """
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(targets * log_probs, axis=-1)
    
    if weights is not None:
        loss = loss * weights
        return jnp.sum(loss) / jnp.maximum(jnp.sum(weights), 1.0)
    
    return jnp.mean(loss) 
"""Loss functions related to predicted aligned error (PAE).

This module contains functions for calculating losses based on the
Predicted Aligned Error (PAE) matrix, which represents the expected error
in the predicted positions between pairs of residues.
"""

import jax
import jax.numpy as jnp
import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Tuple, Union, Any

from alphafold3.design.losses import common


def get_pae_matrix(result):
    """Extract PAE matrix from model result, handling different formats.
    
    Args:
        result: AlphaFold model result dictionary or confidence head output.
        
    Returns:
        PAE matrix as a JAX array with shape (R, R), or None if PAE not found.
    """
    pae = None
    if isinstance(result, dict):
        # Check various possible PAE keys
        for pae_key in ['full_pae', 'pae']:  # Prioritize 'full_pae' if available
            if pae_key in result:
                pae = result[pae_key]
                break
        else:
            logging.warning("Could not find PAE data in result. Format may be incorrect.")
            return None
    else:
        logging.warning("Result is not a dictionary. Cannot extract PAE.")
        return None
    
    return jnp.asarray(pae)  # Ensure it's a JAX array


def process_pae_dimensions(pae):
    """Process PAE matrix to ensure it has shape (R, R), handling batch dimensions.
    
    Args:
        pae: PAE matrix as a JAX array with potential batch dimensions.
        
    Returns:
        Processed PAE matrix with shape (R, R).
    """
    if pae is None:
        return jnp.array(0.0, dtype=jnp.float32)
        
    # Use Python conditional on rank (static property)
    if pae.ndim > 2:
        # Average over the first (batch/sample) dimension
        pae_processed = jnp.mean(pae, axis=0) 
    elif pae.ndim == 2:
        pae_processed = pae # Already 2D
    else:
        # Should not happen for PAE, return zero or raise error
        logging.error(f"Unexpected PAE rank: {pae.ndim}")
        return jnp.array(0.0, dtype=jnp.float32)

    # Ensure output is 2D
    if pae_processed.ndim != 2:
         logging.error(f"PAE processing resulted in wrong rank: {pae_processed.ndim}")
         return jnp.array(0.0, dtype=jnp.float32)

    return pae_processed


def get_interface_pae_loss(result, target_indices, binder_indices):
    """Calculate loss based on PAE values at the interface.

    Args:
        result: AlphaFold model result dictionary or confidence head output.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.

    Returns:
        Average PAE at the interface (target-binder interactions).
    """
    logger.debug(
        f"get_interface_pae_loss called with keys={list(result.keys())}, target_indices.shape={getattr(target_indices,'shape',None)}, binder_indices.shape={getattr(binder_indices,'shape',None)}"
    )
    # Extract and process PAE matrix
    pae = get_pae_matrix(result)
    if pae is None:
        return jnp.array(0.0, dtype=jnp.float32)
        
    pae_processed = process_pae_dimensions(pae)
    
    # Ensure indices are JAX arrays
    target_indices_jax = jnp.asarray(target_indices)
    binder_indices_jax = jnp.asarray(binder_indices)

    # --- JAX-friendly check for 2D shape before indexing ---
    def index_pae_2d(arr_2d):
        interface_pae = arr_2d[jnp.ix_(target_indices_jax, binder_indices_jax)]
        # Use nan_to_num before mean for safety
        return jnp.mean(jnp.nan_to_num(interface_pae))

    def handle_non_2d_pae(arr_non_2d):
        return jnp.array(0.0, dtype=jnp.float32)

    pae_loss = jax.lax.cond(
        jnp.equal(pae_processed.ndim, 2),
        index_pae_2d,
        handle_non_2d_pae,
        pae_processed
    )

    logger.debug(f"get_interface_pae_loss returning loss: {pae_loss}")
    return pae_loss


def get_masked_pae_loss(result, row_indices, col_indices):
    """Calculate loss based on PAE values for any specified sets of residue pairs.

    Args:
        result: AlphaFold model result dictionary or confidence head output.
        row_indices: Row indices for the PAE matrix.
        col_indices: Column indices for the PAE matrix.

    Returns:
        Average PAE for the specified pairs of residues.
    """
    # Extract and process PAE matrix
    pae = get_pae_matrix(result)
    if pae is None:
        return jnp.array(0.0, dtype=jnp.float32)
        
    pae_processed = process_pae_dimensions(pae)
    
    # Mask the PAE matrix to focus on the specified residue pairs
    masked_pae = common.mask_interfaces(pae_processed, row_indices, col_indices)
    
    # Calculate the mean of the masked PAE values
    return common.safe_mean(masked_pae)


def get_chain_pae_loss(result, chain_indices):
    """Calculate loss based on internal PAE values within a chain.

    Args:
        result: AlphaFold model result dictionary or confidence head output.
        chain_indices: Indices of the residues in the chain.

    Returns:
        Average internal PAE for the specified chain.
    """
    return get_masked_pae_loss(result, chain_indices, chain_indices)


def calculate_pae_loss(
    result: Dict[str, Any],
    binder_indices: Optional[jnp.ndarray] = None,
    target_indices: Optional[jnp.ndarray] = None,
    interface_only: bool = True,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Calculate loss based on PAE scores.
    
    Args:
        result: Dictionary of model outputs including predicted_aligned_error
        binder_indices: Indices of the binder residues
        target_indices: Indices of the target residues
        interface_only: If True, only consider PAE at the interface
        scale: Scaling factor for the loss
    
    Returns:
        PAE-based loss value
    """
    logger.debug(
        f"calculate_pae_loss called with result keys={list(result.keys())}, binder_indices={binder_indices}, target_indices={target_indices}, interface_only={interface_only}, scale={scale}"
    )
    if "predicted_aligned_error" not in result:
        raise ValueError("No predicted_aligned_error in result dict")
    
    pae = result["predicted_aligned_error"]  # Shape: [N, N]
    
    if interface_only and (binder_indices is not None) and (target_indices is not None):
        # Focus on interface PAE
        pae_masked = common.mask_interfaces(pae, binder_indices, target_indices)
    else:
        pae_masked = pae
    
    # Loss is mean PAE (we want to minimize PAE)
    loss = common.safe_mean(pae_masked) * scale
    
    logger.debug(f"calculate_pae_loss returning loss: {loss}")
    return loss

def calculate_pae_confidence_loss(
    result: Dict[str, Any],
    binder_indices: jnp.ndarray,
    target_indices: jnp.ndarray,
    confidence_threshold: float = 0.8,
) -> jnp.ndarray:
    """Calculate PAE loss focused on high-confidence interface predictions.
    
    Args:
        result: Dictionary of model outputs
        binder_indices: Indices of the binder residues
        target_indices: Indices of the target residues
        confidence_threshold: Confidence threshold for including residue pairs
    
    Returns:
        PAE confidence-based loss value
    """
    logger.debug(
        f"calculate_pae_confidence_loss called with result keys={list(result.keys())}, binder_indices={binder_indices}, target_indices={target_indices}, confidence_threshold={confidence_threshold}"
    )
    if "predicted_aligned_error" not in result or "predicted_aligned_error_confidence" not in result:
        raise ValueError("Missing predicted_aligned_error or predicted_aligned_error_confidence in result dict")
    
    pae = result["predicted_aligned_error"]  # Shape: [N, N]
    confidence = result["predicted_aligned_error_confidence"]  # Shape: [N, N]
    
    # Create interface mask
    interface_pae = common.mask_interfaces(pae, binder_indices, target_indices)
    interface_conf = common.mask_interfaces(confidence, binder_indices, target_indices)
    
    # Apply confidence threshold
    high_conf_mask = interface_conf >= confidence_threshold
    high_conf_pae = jnp.where(high_conf_mask, interface_pae, jnp.nan)
    
    # Loss is mean PAE for high-confidence predictions
    loss = common.safe_mean(high_conf_pae)
    
    logger.debug(f"calculate_pae_confidence_loss returning loss: {loss}")
    return loss

def calculate_max_interface_pae_loss(
    result: Dict[str, Any],
    binder_indices: jnp.ndarray,
    target_indices: jnp.ndarray,
    percentile: float = 90.0,
) -> jnp.ndarray:
    """Calculate PAE loss based on high percentile PAE values at the interface.
    
    Args:
        result: Dictionary of model outputs
        binder_indices: Indices of the binder residues
        target_indices: Indices of the target residues
        percentile: Percentile of PAE values to use for loss
    
    Returns:
        PAE percentile-based loss value
    """
    logger.debug(
        f"calculate_max_interface_pae_loss called with result keys={list(result.keys())}, binder_indices={binder_indices}, target_indices={target_indices}, percentile={percentile}"
    )
    if "predicted_aligned_error" not in result:
        raise ValueError("No predicted_aligned_error in result dict")
    
    pae = result["predicted_aligned_error"]  # Shape: [N, N]
    
    # Create interface mask
    interface_pae = common.mask_interfaces(pae, binder_indices, target_indices)
    
    # Flatten and check for validity
    flat_pae = interface_pae.flatten()
    is_valid = jnp.isfinite(flat_pae)
    num_valid = jnp.sum(is_valid)

    def calculate_loss(x):
        # Choose value to replace NaN based on percentile
        replacement_val = jax.lax.cond(jnp.equal(percentile, 100.0),
                                       lambda _: -jnp.inf, 
                                       lambda _: jnp.inf,
                                       None)
        
        valid_vals = jnp.where(is_valid, x, replacement_val)

        # Compute percentile or max
        loss_val = jax.lax.cond(jnp.equal(percentile, 100.0),
                                lambda v: jnp.max(v), # Max ignores -inf
                                lambda v: jnp.percentile(v, percentile), # Percentile ignores inf
                                valid_vals)
        return loss_val

    # Calculate loss only if there are valid PAE values
    loss = jax.lax.cond(
        num_valid > 0,
        calculate_loss,
        lambda x: jnp.array(0.0), # Return 0 if no valid entries
        flat_pae
    )

    logger.debug(f"calculate_max_interface_pae_loss returning loss: {loss}")
    return loss 
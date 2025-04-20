"""Loss functions based on pLDDT (predicted local distance difference test) scores."""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional, Union, Tuple

from alphafold3.design.losses.common import safe_mean


def get_binder_plddt_loss(
    result: Dict[str, Any], 
    binder_indices: jnp.ndarray
) -> jnp.ndarray:
    """Calculate loss based on pLDDT values for binder residues.

    Args:
        result: AlphaFold model result dictionary or confidence head output.
        binder_indices: Indices of the binder residues.

    Returns:
        Negative average pLDDT (to minimize as a loss function).
    """
    # Handle different output formats
    plddt = None
    if isinstance(result, dict) and 'predicted_lddt' in result:
        plddt = result['predicted_lddt']
    else:
        logging.warning("Could not find predicted_lddt in result. Format may be incorrect.")
        # Return a JAX array to maintain type consistency for JAX transformations
        return jnp.array(0.0, dtype=jnp.float32)

    # Ensure plddt is a JAX array for consistent operations
    plddt = jnp.asarray(plddt)
    binder_indices = jnp.asarray(binder_indices)  # Ensure indices are JAX array

    # --- Simplified Shape Handling: Assume Rank 3 (samples, residues, atoms) ---
    # If plddt might sometimes be rank 2 (residues, atoms), add a check or handle appropriately.
    try:
        # Correct indexing: axis 1 for residues
        # binder_indices should be valid for the padded dimension
        binder_plddt_samples_atoms = plddt[:, binder_indices, :]  # Assumes rank 3

        # Average over atoms (last axis, -1) and samples (first axis, 0)
        binder_plddt_mean_per_residue = jnp.mean(binder_plddt_samples_atoms, axis=(0, -1))

    except IndexError as e:
         # This might occur if plddt has fewer than 3 dimensions
         logging.error(f"Error indexing pLDDT (shape={plddt.shape}) assuming rank 3: {e}. Returning 0 loss.")
         return jnp.array(0.0, dtype=jnp.float32)
    except Exception as e:
         # Catch other potential errors during calculation
         logging.error(f"Error calculating binder pLDDT mean (shape={plddt.shape}): {e}. Returning 0 loss.")
         return jnp.array(0.0, dtype=jnp.float32)
    # --- End Simplified Shape Handling ---

    # Return negative average pLDDT (as we want to maximize pLDDT)
    # Ensure the result is scalar
    # Use nan_to_num for safety before final mean, although mean usually handles NaNs
    return -jnp.mean(jnp.nan_to_num(binder_plddt_mean_per_residue))

def calculate_plddt_loss(
    result: Dict[str, Any],
    selection_indices: Optional[jnp.ndarray] = None,
    threshold: float = 70.0,
    mode: str = "maximize",
) -> jnp.ndarray:
    """Calculate loss based on pLDDT scores.
    
    Args:
        result: Dictionary of model outputs including plddt scores
        selection_indices: Optional indices to select specific residues
        threshold: pLDDT threshold for optimization (default: 70.0)
        mode: Either "maximize" to increase pLDDT or "threshold" to 
              optimize towards the threshold value
              
    Returns:
        pLDDT-based loss value
    """
    if "predicted_lddt" not in result:
        logging.warning("calculate_plddt_loss: 'predicted_lddt' not found in result.")
        return jnp.array(0.0, dtype=jnp.float32)
    plddt = jnp.asarray(result["predicted_lddt"]) # Use correct key

    # Handle potential multiple dimensions (e.g., sample, atoms)
    if plddt.ndim > 1:
        plddt = jnp.mean(plddt, axis=tuple(range(plddt.ndim))[1:]) # Mean over all except residue dim

    # Apply selection mask if specified
    if selection_indices is not None:
        mask = jnp.zeros_like(plddt, dtype=jnp.bool_)
        mask = mask.at[selection_indices].set(True)
        plddt_selected = jnp.where(mask, plddt, jnp.nan)
    else:
        plddt_selected = plddt
    
    if mode == "maximize":
        # Loss is negative mean pLDDT (we want to maximize pLDDT)
        loss = -safe_mean(plddt_selected)
    elif mode == "threshold":
        # Loss is mean squared difference from threshold
        loss = safe_mean(jnp.square(plddt_selected - threshold))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return loss

def calculate_plddt_confidence_weighted_loss(
    result: Dict[str, Any],
    confidence_weights: jnp.ndarray,
    selection_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Calculate pLDDT loss weighted by confidence scores.
    
    Args:
        result: Dictionary of model outputs including plddt scores
        confidence_weights: Array of confidence weights for each residue
        selection_indices: Optional indices to select specific residues
        
    Returns:
        Confidence-weighted pLDDT loss
    """
    if "predicted_lddt" not in result:
        logging.warning("calculate_plddt_confidence_weighted_loss: 'predicted_lddt' not found in result.")
        return jnp.array(0.0, dtype=jnp.float32)
    plddt = jnp.asarray(result["predicted_lddt"]) # Correct key and ensure JAX array

    # Ensure plddt is at least 1D (handle potential scalar case)
    if plddt.ndim == 0:
        plddt = plddt[None] # Promote to 1D
        
    # Assume plddt might have sample/atom dimensions -> average them out
    # This aligns with get_binder_plddt_loss logic implicitly handling multiple dims
    if plddt.ndim > 1:
        plddt = jnp.mean(plddt, axis=tuple(range(plddt.ndim))[1:]) # Mean over all except residue dim

    # Ensure confidence_weights matches the residue dimension
    if confidence_weights.shape != plddt.shape:
        logging.warning(f"Shape mismatch: plddt {plddt.shape}, confidence_weights {confidence_weights.shape}")
        # Attempt to broadcast or handle error gracefully - for now, return 0
        # This might need refinement based on expected input shapes
        # For testing, we assume shapes match after averaging plddt
        # Fallback if shapes still don't match:
        if confidence_weights.shape != plddt.shape:
           return jnp.array(0.0, dtype=jnp.float32)

    # Apply selection mask if specified
    if selection_indices is not None:
        mask = jnp.zeros_like(plddt, dtype=jnp.bool_)
        mask = mask.at[selection_indices].set(True)
        
        # Apply mask to pLDDT and weights
        plddt_selected = jnp.where(mask, plddt, 0.0)
        weights_selected = jnp.where(mask, confidence_weights, 0.0)
        
        # Normalize weights
        sum_weights = jnp.sum(weights_selected)
        weights_selected = jnp.where(sum_weights > 0, 
                                    weights_selected / sum_weights, 
                                    weights_selected)
        
        # Weighted negative mean (to maximize)
        loss = -jnp.sum(plddt_selected * weights_selected)
    else:
        # Normalize weights
        sum_weights = jnp.sum(confidence_weights)
        norm_weights = jnp.where(sum_weights > 0, 
                                confidence_weights / sum_weights, 
                                confidence_weights)
        
        # Weighted negative mean (to maximize)
        loss = -jnp.sum(plddt * norm_weights)
    
    return loss 
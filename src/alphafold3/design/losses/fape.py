"""Loss functions related to frame alignment (FAPE)."""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional, Tuple

from alphafold3.design.losses.common import safe_mean

# Moved from binder_loss.py
def get_target_fape_loss(result, initial_coords, target_indices):
    """Calculate loss to maintain target structure using FAPE-like RMSD.

    Args:
        result: AlphaFold model result dictionary.
        initial_coords: Initial coordinates of the structure (NumPy or JAX array).
        target_indices: Indices of the target residues (NumPy or JAX array).

    Returns:
        FAPE-like loss for target residues.
    """
    logging.debug(
        f"get_target_fape_loss called with result keys={list(result.keys())}, "
        f"initial_coords shape={getattr(initial_coords, 'shape', None)}, "
        f"target_indices={getattr(target_indices, 'shape', None)}"
    )
    # Get final coordinates from result
    if 'final_atom_positions' not in result:
        logging.warning("final_atom_positions not found in result for FAPE calculation.")
        return jnp.array(0.0, dtype=jnp.float32)

    final_coords = jnp.asarray(result['final_atom_positions'])
    initial_coords = jnp.asarray(initial_coords) # Ensure JAX array
    target_indices = jnp.asarray(target_indices) # Ensure JAX array
    logging.debug(
        f"final_coords.shape={final_coords.shape}, "
        f"initial_coords.shape={initial_coords.shape}, "
        f"target_indices.shape={target_indices.shape}"
    )

    # Add checks for compatible shapes before indexing - These Python checks might fail under JIT/grad
    # It's safer to perform shape checks outside the JITted function or handle potential errors inside.
    # For now, assume shapes are compatible based on typical usage.

    try:
        # Select coordinates for target residues
        # Assuming coordinates shape is (..., num_residues_padded, num_atoms, 3)
        target_initial_coords = initial_coords[..., target_indices, :, :]
        target_final_coords = final_coords[..., target_indices, :, :]
        logging.debug(
            f"target_initial_coords.shape={target_initial_coords.shape}, "
            f"target_final_coords.shape={target_final_coords.shape}"
        )

        # Create mask for valid atoms (non-zero coordinates)
        valid_mask_initial = jnp.any(target_initial_coords != 0, axis=-1)
        valid_mask_final = jnp.any(target_final_coords != 0, axis=-1)
        valid_mask = valid_mask_initial & valid_mask_final

        # Calculate squared distances, applying the mask
        squared_diff = jnp.sum(
            jnp.square(target_final_coords - target_initial_coords) * valid_mask[..., None],
            axis=-1 # Sum over xyz coordinates
        )
        logging.debug(f"squared_diff shape={squared_diff.shape}")

        # Calculate mean squared error, considering only valid atoms
        sum_sq_diff = jnp.sum(squared_diff)
        num_valid_atoms = jnp.sum(valid_mask)
        logging.debug(f"sum_sq_diff={sum_sq_diff}, num_valid_atoms={num_valid_atoms}")
        safe_denom = jnp.maximum(num_valid_atoms, 1e-8)
        safe_rmsd_like_loss = sum_sq_diff / safe_denom
        logging.debug(f"safe_rmsd_like_loss={safe_rmsd_like_loss}")

        # Use nan_to_num for final safety
        return jnp.nan_to_num(safe_rmsd_like_loss)

    except Exception as e:
        # Catch potential shape or indexing errors during JAX operations
        logging.error(f"Error during FAPE calculation (shapes final:{final_coords.shape}, initial:{initial_coords.shape}, indices:{target_indices.shape}): {e}")
        return jnp.array(0.0, dtype=jnp.float32)

# Placeholder - Functions will be moved here 
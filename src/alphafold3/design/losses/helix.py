import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional, Tuple

# Assuming 'safe_mean' is available in your common loss utilities
# Please adjust the import path if necessary
from alphafold3.design.losses.common import safe_mean

logger = logging.getLogger(__name__)

# --- Constants ---
HELIX_CUTOFF_ANGSTROM = 6.0
HELIX_OFFSET = 3 # Look for i, i+3 contacts

# --- Helper Functions ---

def _calculate_binary_contact_loss(
    distogram_logits: jnp.ndarray, # Shape (..., N, N, B)
    bin_edges: jnp.ndarray,        # Shape (B-1,)
    cutoff_angstrom: float
) -> jnp.ndarray:
    """Calculates binary contact cross-entropy based on distogram probabilities.

    Computes -log(P(distance < cutoff)) for each residue pair.

    Args:
        distogram_logits: Raw logits from the distogram head.
        bin_edges: Bin edges for the distogram.
        cutoff_angstrom: The distance cutoff.

    Returns:
        A matrix (..., N, N) of per-pair binary contact losses.
    """
    # Ensure logits are 3D (N, N, B)
    logits_3d = distogram_logits
    if logits_3d.ndim == 4: # Handle potential batch dimension
        logits_3d = logits_3d[0]
    elif logits_3d.ndim != 3:
        raise ValueError(f"Unexpected distogram_logits shape: {distogram_logits.shape}")

    # Add a large value to the first bin edge to ensure correct comparison
    # Bin edges define the *upper* bound of the bins, except the last one.
    # We want bins where the upper bound is < cutoff.
    effective_bin_edges = jnp.concatenate([jnp.array([-jnp.inf]), bin_edges], axis=0) # Shape (B,)

    # Mask for bins representing distances < cutoff
    # Bins are indexed 0 to B-1. Bin i corresponds to distance range [edge_{i-1}, edge_i].
    # We want bins where edge_i < cutoff.
    bins_below_cutoff = effective_bin_edges < cutoff_angstrom # Shape (B,)

    # Calculate probabilities P(bin)
    probs = jax.nn.softmax(logits_3d, axis=-1) # Shape (N, N, B)

    # Sum probabilities of bins below cutoff: P(distance < cutoff)
    prob_dist_lt_cutoff = jnp.sum(probs * bins_below_cutoff[None, None, :], axis=-1) # Shape (N, N)

    # Calculate binary cross-entropy: -log(P(distance < cutoff))
    # Add epsilon for numerical stability
    binary_contact_loss = -jnp.log(prob_dist_lt_cutoff + 1e-8) # Shape (N, N)

    return binary_contact_loss

# --- Main Loss Function ---

def calculate_helix_loss(
    distogram_logits: jnp.ndarray, # Shape (..., N, N, B)
    bin_edges: jnp.ndarray,        # Shape (B-1,)
    residue_index: jnp.ndarray,    # Shape (N,) - Absolute residue indices
    binder_indices: jnp.ndarray,   # Shape (Lb,) - Indices within the N dimension
) -> jnp.ndarray:
    """Calculates a helix propensity loss based on i, i+3 contacts within the binder.

    Uses binary cross-entropy for contacts < 6 Angstroms.

    Args:
        distogram_logits: Raw logits from the distogram head.
        bin_edges: Bin edges for the distogram.
        residue_index: Array of residue indices for offset calculation.
        binder_indices: Indices corresponding to binder residues.

    Returns:
        Scalar helix loss averaged over relevant binder pairs.
    """
    if binder_indices.shape[0] == 0:
        logger.debug("Binder indices are empty, returning 0 helix loss.")
        return jnp.array(0.0, dtype=distogram_logits.dtype)

    num_residues = distogram_logits.shape[-2] # N

    # 1. Calculate per-pair binary contact loss for d < HELIX_CUTOFF_ANGSTROM
    # Shape (N, N)
    pair_contact_loss = _calculate_binary_contact_loss(
        distogram_logits, bin_edges, HELIX_CUTOFF_ANGSTROM
    )

    # 2. Create mask for i, i+3 contacts
    offset = residue_index[:, None] - residue_index[None, :] # Shape (N, N)
    helix_offset_mask = (offset == HELIX_OFFSET) # Shape (N, N)

    # 3. Create mask for pairs where the *first* residue (i) is in the binder
    # This implicitly ensures i+3 is also checked against the binder region
    # if the offset is exactly 3.
    is_binder = jnp.zeros(num_residues, dtype=bool).at[binder_indices].set(True)
    binder_i_mask = is_binder[:, None] # Shape (N, 1), broadcast to (N, N)

    # 4. Combine masks: We want pairs (i, j) where j=i+3 AND i is a binder residue
    final_mask = helix_offset_mask & binder_i_mask # Shape (N, N)

    # 5. Calculate mean loss over the masked pairs
    helix_loss = safe_mean(pair_contact_loss, where=final_mask)

    logger.debug(f"Helix loss calculation: Mask sum = {jnp.sum(final_mask)}, Mean loss = {helix_loss}")

    return helix_loss
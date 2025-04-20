"""Loss functions related to distogram predictions."""

import jax
import jax.numpy as jnp
import functools
from typing import Dict, Any, Optional, Tuple

from alphafold3.design.losses.common import entropy_low_bins, safe_mean

# Moved from binder_loss.py
def get_distogram_entropy_loss(
    distogram_logits, bin_breaks, target_indices, binder_indices, weights
):
    """Calculate loss based on distogram entropy (Paper Eq 2-5 + k-min agg).

    Args:
        distogram_logits: Raw distogram logits (R, R, num_bins).
        bin_breaks: Bin edges (num_bins - 1,).
        target_indices: Indices of the target residues (1D JAX array).
        binder_indices: Indices of the binder residues (1D JAX array).
        weights: Dictionary containing 'contact_intra' and 'contact_inter' weights.

    Returns:
        Combined weighted distogram entropy loss (scalar JAX array).
        Dictionary containing unweighted 'contact_intra' and 'contact_inter' losses.
    """
    # Ensure inputs are JAX arrays
    distogram_logits = jnp.asarray(distogram_logits)
    bin_breaks = jnp.asarray(bin_breaks)
    target_indices = jnp.asarray(target_indices)
    binder_indices = jnp.asarray(binder_indices)
    num_binder = binder_indices.shape[0]
    zero_loss_float32 = jnp.array(0.0, dtype=jnp.float32)

    # --- Pre-compute helpers ---
    # Calculate bin tops needed for cutoff masks
    bin_tops = jnp.append(bin_breaks, bin_breaks[-1] + (bin_breaks[-1] - bin_breaks[-2])) # Shape (num_bins,)

    # Create masks based on cutoffs
    low_bin_mask_14 = (bin_tops <= 14.0) # Intra-binder cutoff
    low_bin_mask_22 = (bin_tops <= 22.0) # Inter-interface cutoff

    # Create partial functions for entropy calculation with specific masks
    ent_intra_fn = functools.partial(entropy_low_bins, low_mask=low_bin_mask_14)
    ent_inter_fn = functools.partial(entropy_low_bins, low_mask=low_bin_mask_22)

    # --- Calculate per-pair losses ---
    # Calculate entropy for all pairs using both cutoffs (vectorized)
    # We could optimize by slicing logits first, but this is simpler for now.
    # Assuming logits shape (R, R, 64)
    intra_pair_entropy = ent_intra_fn(distogram_logits) # Shape (R, R)
    inter_pair_entropy = ent_inter_fn(distogram_logits) # Shape (R, R)

    # --- Residue-level aggregation ---

    # Intra-binder loss (k=2, |i-j|>=9)
    intra_loss = zero_loss_float32
    w_intra = jnp.asarray(weights.get('contact_intra', 0.0)) # Get weight

    def calc_intra_loss():
        # Select intra-binder pairs
        ib_entropy = intra_pair_entropy[jnp.ix_(binder_indices, binder_indices)] # Shape (num_binder, num_binder)

        # Create separation mask (|i-j| >= 9)
        binder_coords = jnp.arange(num_binder)
        sep_mask = jnp.abs(binder_coords[:, None] - binder_coords[None, :]) >= 9

        # Apply mask: set invalid pairs' entropy to infinity for sorting
        ib_entropy_masked = jnp.where(sep_mask, ib_entropy, jnp.inf)

        # Sort entropies for each residue and pick the two smallest (k=2)
        ib_sorted = jnp.sort(ib_entropy_masked, axis=-1) # Sort along the second dimension

        # --- JAX-compatible k-min mean using masking ---
        def pick_k_or_less(row):
            # row is already sorted, finite values come first
            k_intra = 2 # Static value

            # Create a mask for the first k_intra elements
            indices = jnp.arange(row.shape[0])
            mask = indices < k_intra

            # Also mask non-finite values (like the inf used for sorting)
            mask = mask & jnp.isfinite(row)

            # Sum the masked elements and count them
            masked_sum = jnp.sum(jnp.where(mask, row, 0.0))
            masked_count = jnp.sum(mask) # Counts only True values in the final mask

            # Calculate mean safely, return 0 if count is 0
            mean_val = jnp.where(masked_count > 0, masked_sum / masked_count, 0.0)
            return mean_val
        # --- End JAX-compatible k-min mean ---

        # Apply this row-wise using vmap
        per_res_intra_mean_k = jax.vmap(pick_k_or_less)(ib_sorted) # Shape (num_binder,)

        # Final intra-loss: mean over all binder residues (l = num_binder)
        # Use nan_to_num for safety, though mean should handle empty cases if num_binder > 0
        return jnp.mean(jnp.nan_to_num(per_res_intra_mean_k))

    # Conditionally calculate intra loss based on weight and num_binder
    intra_loss = jax.lax.cond(
        (w_intra > 0.0) & (num_binder > 0),
        lambda _: calc_intra_loss(),
        lambda _: zero_loss_float32,
        None # No operand needed
    )

    # Inter-face loss (k=1)
    inter_loss = zero_loss_float32
    w_inter = jnp.asarray(weights.get('contact_inter', 0.0)) # Get weight

    def calc_inter_loss():
        # Select inter-face pairs (binder-target)
        it_entropy = inter_pair_entropy[jnp.ix_(binder_indices, target_indices)] # Shape (num_binder, num_target)

        # For each binder residue, pick the minimum entropy contact (k=1)
        # jnp.min automatically handles empty arrays if target_indices is empty (returns inf)
        # Use nan_to_num just in case entropy calculation resulted in NaN somewhere
        per_res_inter_min = jnp.min(jnp.nan_to_num(it_entropy, nan=jnp.inf), axis=-1) # Shape (num_binder,)

        # Final inter-loss: mean over all binder residues
        # Filter out inf values that result from rows with no target contacts or all NaNs/Infs
        per_res_inter_finite = jnp.where(jnp.isfinite(per_res_inter_min), per_res_inter_min, 0.0)
        finite_count = jnp.sum(jnp.isfinite(per_res_inter_min))
        # Avoid division by zero if no finite values exist
        mean_inter_loss = jnp.where(finite_count > 0, jnp.sum(per_res_inter_finite) / finite_count, 0.0)
        return mean_inter_loss

    # Conditionally calculate inter loss based on weight and num_binder/num_target
    num_target = target_indices.shape[0]
    inter_loss = jax.lax.cond(
        (w_inter > 0.0) & (num_binder > 0) & (num_target > 0),
        lambda _: calc_inter_loss(),
        lambda _: zero_loss_float32,
        None # No operand needed
    )

    # --- Combine weighted losses ---
    total_distogram_loss = w_intra * intra_loss + w_inter * inter_loss

    # Return breakdown (unweighted individual losses) and total weighted loss
    loss_breakdown = {'contact_intra': intra_loss, 'contact_inter': inter_loss}
    return total_distogram_loss, loss_breakdown 
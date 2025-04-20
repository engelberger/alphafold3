"""Gradient-based loss functions for protein design."""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional, Tuple

from alphafold3.design.losses import plddt, pae, distogram, sequence, fape
from alphafold3.design.losses.common import safe_mean

logger = logging.getLogger(__name__)


def calculate_gradient_binder_loss(
    result: Dict[str, Any],
    feature_dict: Dict[str, Any],
    target_indices: jnp.ndarray,
    binder_indices: jnp.ndarray,
    binder_seq_logits: jnp.ndarray,
    design_params: Dict[str, Any],
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Calculate loss for gradient-based binder design.

    Combines various loss components based on weights specified in design_params.

    Args:
        result: AlphaFold model result dictionary.
        feature_dict: Feature dictionary.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.
        binder_seq_logits: Sequence logits for the binder.
        design_params: Dictionary of design parameters (containing weights).

    Returns:
        Tuple of (total_loss, loss_breakdown).
    """
    losses = {}
    weights = design_params.get("weights", {}) # Get weights dict
    zero_loss_float32 = jnp.array(0.0, dtype=jnp.float32)

    # Ensure indices are JAX arrays
    target_indices_jax = jnp.asarray(target_indices)
    binder_indices_jax = jnp.asarray(binder_indices)

    # --- Calculate pLDDT Loss ---
    plddt_weight = weights.get('plddt', 0.0)
    losses['plddt'] = jax.lax.cond(
        jnp.greater(jnp.asarray(plddt_weight), 0.0),
        lambda op: plddt.get_binder_plddt_loss(op[0], op[1]), # Use lambda with operand tuple
        lambda op: zero_loss_float32,
        (result, binder_indices_jax) # Pass operands as tuple
    )

    # --- Calculate Interface PAE Loss ---
    pae_inter_weight = weights.get('pae_inter', 0.0)
    losses['pae_inter'] = jax.lax.cond(
        jnp.greater(jnp.asarray(pae_inter_weight), 0.0),
        lambda op: pae.get_interface_pae_loss(op[0], op[1], op[2]),
        lambda op: zero_loss_float32,
        (result, target_indices_jax, binder_indices_jax) # Pass operands as tuple
    )

    # --- Calculate Distogram Entropy Loss (New) ---
    # Check if required weights exist and are > 0
    w_intra = weights.get('contact_intra', 0.0)
    w_inter = weights.get('contact_inter', 0.0)
    distogram_pred = jnp.greater(jnp.asarray(w_intra) + jnp.asarray(w_inter), 0.0)

    def calc_distogram_true(operand):
        res, tgt_idx, bnd_idx, w = operand
        # Check if distogram results are present
        if 'distogram' in res and isinstance(res['distogram'], dict) and \
           'logits' in res['distogram'] and 'bin_edges' in res['distogram']:
            total_dist_loss, breakdown = distogram.get_distogram_entropy_loss(
                distogram_logits=res['distogram']['logits'],
                bin_breaks=res['distogram']['bin_edges'],
                target_indices=tgt_idx,
                binder_indices=bnd_idx,
                weights=w # Pass weights dict directly
            )
            # Return the breakdown and the total weighted loss
            return breakdown.get('contact_intra', zero_loss_float32), \
                   breakdown.get('contact_inter', zero_loss_float32), \
                   total_dist_loss
        else:
            logger.warning("Distogram logits or bin_edges missing, cannot calculate entropy loss.")
            return zero_loss_float32, zero_loss_float32, zero_loss_float32

    def calc_distogram_false(_):
        return zero_loss_float32, zero_loss_float32, zero_loss_float32

    # Calculate distogram losses conditionally
    loss_intra, loss_inter, weighted_distogram_loss = jax.lax.cond(
        distogram_pred,
        calc_distogram_true,
        calc_distogram_false,
        operand=(result, target_indices_jax, binder_indices_jax, weights) # Pass operands
    )
    losses['contact_intra'] = loss_intra # Store unweighted intra loss
    losses['contact_inter'] = loss_inter # Store unweighted inter loss
    # Note: total_loss below will add the *already weighted* distogram_loss

    # --- Calculate Sequence Entropy Loss ---
    seq_entropy_weight = weights.get('seq_entropy', 0.0)
    losses['seq_entropy'] = jax.lax.cond(
        jnp.greater(jnp.asarray(seq_entropy_weight), 0.0),
        lambda logits: sequence.get_binder_seq_entropy_loss(logits),
        lambda logits: zero_loss_float32,
        binder_seq_logits # Operand
    )

    # --- Calculate Target FAPE Loss ---
    fape_target_weight = weights.get('fape_target', 0.0)
    # This condition also depends on feature_dict key presence, handled inside the 'true' function
    def calc_fape_true(operand):
        res, feat_dict, tgt_idx = operand
        # Python dictionary key check is okay here before JAX ops
        if 'initial_coords' in feat_dict:
             return fape.get_target_fape_loss(res, feat_dict['initial_coords'], tgt_idx)
        else:
             # Avoid logging inside cond branches if possible
             logger.warning("FAPE target weight > 0 but 'initial_coords' not in feature_dict.")
             return zero_loss_float32
    def calc_fape_false(_):
        return zero_loss_float32

    losses['fape_target'] = jax.lax.cond(
        jnp.greater(jnp.asarray(fape_target_weight), 0.0),
        calc_fape_true,
        calc_fape_false,
        (result, feature_dict, target_indices_jax) # Pass tuple of operands
    )

    # Calculate weighted total loss
    total_loss = zero_loss_float32
    loss_breakdown_unweighted = {} # Ensure this dict exists
    # Use keys from the initialized losses dict to ensure all are included
    for k in losses.keys():
        loss_val = losses[k]
        if k in ['contact_intra', 'contact_inter']:
             # These are already captured in weighted_distogram_loss, store unweighted only
             loss_breakdown_unweighted[k] = loss_val
             continue # Skip adding them directly to total_loss

        weight_val = weights.get(k, 0.0) # Get corresponding weight
        weighted_loss = jnp.asarray(weight_val) * loss_val # Ensure JAX array multiplication
        total_loss = total_loss + weighted_loss
        loss_breakdown_unweighted[k] = loss_val # Store unweighted loss

    # Add the pre-weighted distogram loss to the total
    total_loss = total_loss + weighted_distogram_loss

    # Ensure the breakdown dict has entries for the distogram losses
    if 'contact_intra' not in loss_breakdown_unweighted:
        loss_breakdown_unweighted['contact_intra'] = losses['contact_intra']
    if 'contact_inter' not in loss_breakdown_unweighted:
        loss_breakdown_unweighted['contact_inter'] = losses['contact_inter']

    return total_loss, loss_breakdown_unweighted 
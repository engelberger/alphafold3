"""BoltzDesign1-like loss functions for protein design."""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional, Tuple
import functools

from alphafold3.design.losses import plddt, pae, distogram, sequence
from alphafold3.design.losses.common import safe_mean, entropy_low_bins, INTRA_CUTOFF_ANGSTROM, INTER_CUTOFF_ANGSTROM

logger = logging.getLogger(__name__)


def calculate_boltzdesign_distogram_loss(
    distogram_logits: jnp.ndarray, # Shape (..., N, N, B)
    bin_edges: jnp.ndarray,        # Shape (B-1,)
    binder_indices: jnp.ndarray,
    target_indices: jnp.ndarray,
    weights: Dict[str, float],
    inter_k: Optional[int] = None, # Override for k-min aggregation
    inter_l: Optional[int] = None, # Override for l-mean aggregation
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Calculates the distogram entropy loss for BoltzDesign.

    Implements the intra-binder and inter-chain contact losses based on
    entropy of low-distance bins, including k/l aggregation and sequence
    separation filters, as described in Bohnuud et al., BioRxiv 2024.

    Args:
        distogram_logits: Raw logits from the distogram head.
        bin_edges: Bin edges for the distogram.
        binder_indices: 1D array of indices corresponding to binder residues.
        target_indices: 1D array of indices corresponding to target residues.
        weights: Dictionary containing loss weights ('contact_intra', 'contact_inter').
        inter_k: Optional override for number of best contacts per residue (default 1).
        inter_l: Optional override for number of best residues to average (default all).

    Returns:
        Tuple of (total_distogram_loss, distogram_loss_breakdown).
    """
    loss_breakdown = {}
    total_loss = 0.0

    # Create partial functions for entropy calculation with specific cutoffs
    ent_intra_fn = functools.partial(entropy_low_bins, bin_edges=bin_edges, cutoff_angstrom=INTRA_CUTOFF_ANGSTROM)
    ent_inter_fn = functools.partial(entropy_low_bins, bin_edges=bin_edges, cutoff_angstrom=INTER_CUTOFF_ANGSTROM)

    # Calculate per-pair entropy matrices
    # Assuming logits shape is (N, N, B) after potential batch dim removal
    logits_2d = distogram_logits
    if logits_2d.ndim == 4: # Handle potential batch dimension
        logits_2d = logits_2d[0]
    elif logits_2d.ndim != 3:
        raise ValueError(f"Unexpected distogram_logits shape: {distogram_logits.shape}")

    intra_pair_entropy = ent_intra_fn(logits_2d) # (N, N)
    inter_pair_entropy = ent_inter_fn(logits_2d) # (N, N)

    w_intra = weights.get('contact_intra', 0.0)
    if w_intra > 0:
        # Intra-binder loss (compactness)
        binder_len = len(binder_indices)
        ib_entropy = intra_pair_entropy[jnp.ix_(binder_indices, binder_indices)] # (Lb, Lb)

        # Apply sequence separation filter (|i-j| >= 9)
        sep_mask = jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :]) >= 9
        ib_entropy_filtered = jnp.where(sep_mask, ib_entropy, jnp.inf) # Mask out close contacts

        # k=2 aggregation: mean of 2 smallest entropy values per residue
        # Sort entropies for each residue i with respect to all other residues j
        ib_sorted = jnp.sort(ib_entropy_filtered, axis=-1) # Sort along the j axis
        # Take the mean of the first two (k=2) smallest entropy values
        per_res_intra_entropy = jnp.mean(ib_sorted[:, :2], axis=-1) # (Lb,)

        # l=binder_length aggregation: mean over all binder residues
        intra_loss = jnp.mean(per_res_intra_entropy)
        total_loss += w_intra * intra_loss
        loss_breakdown['contact_intra'] = intra_loss
    else:
        loss_breakdown['contact_intra'] = 0.0

    w_inter = weights.get('contact_inter', 0.0)
    inter_loss_val = 0.0 # Initialize
    if w_inter > 0:
        # Inter-chain loss (binding interface)
        it_entropy = inter_pair_entropy[jnp.ix_(binder_indices, target_indices)] # (Lb, Lt)

        # Default k=1, l=Lb (mean of min contact per binder residue)
        k = 1
        l = len(binder_indices)

        # Override k, l if provided (specifically for holo phase)
        if inter_k is not None:
            k = inter_k
            logger.debug(f"Using override inter_k={k}")
        if inter_l is not None:
            l = inter_l
            logger.debug(f"Using override inter_l={l}")

        # Ensure k and l are within valid bounds
        num_targets = it_entropy.shape[1]
        num_binders = it_entropy.shape[0]
        k = min(k, num_targets) if num_targets > 0 else 1
        l = min(l, num_binders) if num_binders > 0 else 1

        # Calculate per-binder-residue best k contacts
        # Sort contacts for each binder residue: (Lb, Lt) -> (Lb, Lt)
        it_sorted = jnp.sort(it_entropy, axis=-1)

        # Select top k contacts: (Lb, Lt) -> (Lb, k)
        # Handle case where k > num_targets safely
        top_k_contacts = it_sorted[:, :k]

        # Average the top k contacts for each binder residue: (Lb, k) -> (Lb,)
        per_res_mean_top_k = jnp.mean(top_k_contacts, axis=-1)

        # Select the top l binder residues based on their mean top-k contact entropy
        # Sort binder residues by their score: (Lb,) -> (Lb,)
        binder_scores_sorted = jnp.sort(per_res_mean_top_k)

        # Take the mean of the best l binder residues
        # Handle case where l > num_binders safely
        inter_loss_val = jnp.mean(binder_scores_sorted[:l])

        total_loss += w_inter * inter_loss_val
        loss_breakdown['contact_inter'] = inter_loss_val
    else:
        loss_breakdown['contact_inter'] = 0.0

    return total_loss, loss_breakdown

def calculate_boltzdesign_binder_loss(
    partial_result: Dict[str, Any], # Model output (potentially partial)
    feature_dict: Dict[str, Any],
    target_indices: jnp.ndarray,
    binder_indices: jnp.ndarray,
    seq_representation: jnp.ndarray, # Softmax probs or STE
    design_params: Dict[str, Any],   # Contains weights dict
    compute_confidence_loss: bool = True, # Flag to control confidence loss computation
    inter_k: Optional[int] = None, # Holo phase k override
    inter_l: Optional[int] = None # Holo phase l override
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Calculates the combined loss for the BoltzDesign protocol.

    Includes distogram entropy loss and optionally confidence losses (pLDDT, PAE).

    Args:
        partial_result: Dictionary containing model outputs (must include
                          distogram[logits], predicted_lddt, full_pae).
        feature_dict: Input feature dictionary.
        target_indices: Indices of target residues.
        binder_indices: Indices of binder residues.
        seq_representation: Sequence representation used (softmax probs or STE).
        design_params: Dictionary containing design parameters, including 'weights'.
        compute_confidence_loss: If True, compute and add pLDDT/PAE losses.
                                   Defaults to True for backward compatibility.
        inter_k: Optional override for k in inter-chain distogram loss.
        inter_l: Optional override for l in inter-chain distogram loss.

    Returns:
        Tuple containing the total loss and a dictionary breakdown of loss components.
    """
    logger.debug("==== STARTING BOLTZDESIGN BINDER LOSS CALCULATION ====")
    logger.debug(f"partial_result keys: {list(partial_result.keys())}")
    weights = design_params.get('weights', {})
    logger.debug(f"design_params (weights) keys: {list(weights.keys())}")
    logger.debug(f"Actual Design weights passed: {weights}")
    logger.debug(f"Compute confidence loss: {compute_confidence_loss}")

    total_loss = 0.0
    loss_breakdown = {}

    # --- Distogram Loss --- 
    # Requires distogram logits and bin edges
    if 'distogram' in partial_result and 'logits' in partial_result['distogram'] and 'bin_edges' in partial_result['distogram']:
        distogram_loss, disto_breakdown = calculate_boltzdesign_distogram_loss(
            distogram_logits=partial_result['distogram']['logits'],
            bin_edges=partial_result['distogram']['bin_edges'],
            binder_indices=binder_indices,
            target_indices=target_indices,
            weights=weights,
            inter_k=inter_k,
            inter_l=inter_l
        )
        total_loss += distogram_loss
        loss_breakdown.update(disto_breakdown)
    else:
        logger.warning("Missing distogram logits or bin_edges in partial_result. Skipping distogram loss.")
        loss_breakdown['contact_intra'] = 0.0
        loss_breakdown['contact_inter'] = 0.0

    # --- Confidence Losses (Conditional) --- 
    if compute_confidence_loss:
        w_confidence = weights.get('confidence', 0.0)
        if w_confidence > 0:
            conf_loss = 0.0
            # pLDDT Loss (on binder)
            if 'predicted_lddt' in partial_result:
                plddt = partial_result['predicted_lddt'] # Shape (..., N)
                if plddt.ndim == 2:
                    plddt = plddt[0] # Remove potential batch dim
                binder_plddt = plddt[binder_indices]
                # Use negative pLDDT as loss (minimize -pLDDT -> maximize pLDDT)
                # Scale by confidence weight
                plddt_loss = -jnp.mean(binder_plddt)
                conf_loss += plddt_loss
                loss_breakdown['plddt_neg_mean'] = plddt_loss # Use negative mean as loss component
            else:
                logger.warning("Missing 'predicted_lddt' in partial_result. Skipping pLDDT confidence loss.")
                loss_breakdown['plddt_neg_mean'] = 0.0

            # Interface PAE Loss 
            if 'full_pae' in partial_result:
                pae = partial_result['full_pae'] # Shape (..., N, N)
                if pae.ndim == 3:
                    pae = pae[0]
                interface_pae = pae[jnp.ix_(binder_indices, target_indices)]
                # Use mean interface PAE as loss (minimize PAE)
                pae_loss = jnp.mean(interface_pae)
                conf_loss += pae_loss
                loss_breakdown['pae_inter_mean'] = pae_loss
            else:
                 logger.warning("Missing 'full_pae' in partial_result. Skipping PAE confidence loss.")
                 loss_breakdown['pae_inter_mean'] = 0.0

            # Add weighted confidence loss to total
            total_loss += w_confidence * conf_loss
            loss_breakdown['confidence_unweighted'] = conf_loss # Store unweighted sum for info
        else:
            loss_breakdown['plddt_neg_mean'] = 0.0
            loss_breakdown['pae_inter_mean'] = 0.0
            loss_breakdown['confidence_unweighted'] = 0.0
    else:
        # Ensure keys exist even if not computed
        loss_breakdown['plddt_neg_mean'] = 0.0
        loss_breakdown['pae_inter_mean'] = 0.0
        loss_breakdown['confidence_unweighted'] = 0.0

    # --- Sequence Entropy Loss --- 
    # This is typically added *outside* this function based on the stage (e.g., in BoltzProtocol.loss_fn_for_grad_boltz)
    # If needed here, requires seq_representation and careful handling
    # seq_entropy_weight = weights.get("seq_entropy", 0.0)
    # if seq_entropy_weight > 0:
    #     # Requires seq_representation (probs)
    #     seq_ent_loss = loss_utils.get_binder_seq_entropy_loss(seq_representation)
    #     total_loss += seq_entropy_weight * seq_ent_loss
    #     loss_breakdown['seq_entropy'] = seq_ent_loss

    # --- Helix Loss (Placeholder) --- 
    w_helix = weights.get("helix", 0.0)
    if w_helix > 0:
         # TODO: Implement helix loss calculation
         # Requires secondary structure prediction or residue propensities
         # Example: helix_loss = calculate_helix_loss(partial_result, binder_indices)
         helix_loss = 0.0 # Placeholder
         total_loss += w_helix * helix_loss
         loss_breakdown['helix'] = helix_loss
    else:
         loss_breakdown['helix'] = 0.0

    logger.debug("Completed BoltzDesign loss calculation")
    logger.debug(f"Total Loss: {total_loss}, Breakdown: {loss_breakdown}")
    return total_loss, loss_breakdown 
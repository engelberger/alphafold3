"""BoltzDesign1-like loss functions for protein design."""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional, Tuple
import functools
import dataclasses

from alphafold3.design.losses import plddt, pae, distogram, sequence
from alphafold3.design.losses.common import safe_mean, entropy_low_bins, INTRA_CUTOFF_ANGSTROM, INTER_CUTOFF_ANGSTROM
from alphafold3.design.config import LossWeightsConfig
from alphafold3.design.losses.helix import calculate_helix_loss

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
    logger.debug(
        f"calculate_boltzdesign_distogram_loss called with logits.shape={distogram_logits.shape}, "
        f"bin_edges.shape={bin_edges.shape}, binder_indices={binder_indices.shape}, "
        f"target_indices={target_indices.shape}, weights={weights}, inter_k={inter_k}, inter_l={inter_l}"
    )
    # Debug paper parameters
    logger.debug(f"INTRA_CUTOFF_ANGSTROM={INTRA_CUTOFF_ANGSTROM}, INTER_CUTOFF_ANGSTROM={INTER_CUTOFF_ANGSTROM}")
    # Show sample of bin_edges
    logger.debug(f"bin_edges sample: first5={bin_edges[:5]}, last5={bin_edges[-5:]}")
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
    logger.debug(f"intra_pair_entropy shape={intra_pair_entropy.shape}, sample={intra_pair_entropy.flatten()[:5]}")
    inter_pair_entropy = ent_inter_fn(logits_2d) # (N, N)
    logger.debug(f"inter_pair_entropy shape={inter_pair_entropy.shape}, sample={inter_pair_entropy.flatten()[:5]}")

    w_intra = weights['boltz_contact_intra']
    logger.debug(f"w_intra={w_intra}")
    if w_intra > 0:
        # Intra-binder loss (compactness)
        binder_len = len(binder_indices)
        logger.debug(f"binder_len={binder_len}")
        ib_entropy = intra_pair_entropy[jnp.ix_(binder_indices, binder_indices)] # (Lb, Lb)
        logger.debug(f"ib_entropy shape={ib_entropy.shape}")

        # Apply sequence separation filter (|i-j| >= 9)
        sep_mask = jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :]) >= 9
        logger.debug(f"seq separation mask sum={jnp.sum(sep_mask)}")
        ib_entropy_filtered = jnp.where(sep_mask, ib_entropy, jnp.inf) # Mask out close contacts
        logger.debug(f"ib_entropy_filtered min,max={(jnp.min(ib_entropy_filtered), jnp.max(ib_entropy_filtered))}")

        # k=2 aggregation: mean of 2 smallest entropy values per residue
        # Sort entropies for each residue i with respect to all other residues j
        ib_sorted = jnp.sort(ib_entropy_filtered, axis=-1) # Sort along the j axis
        logger.debug(f"ib_sorted shape={ib_sorted.shape}")

        # Take the mean of the first two (k=2) smallest entropy values
        per_res_intra_entropy = jnp.mean(ib_sorted[:, :2], axis=-1) # (Lb,)
        logger.debug(f"per_res_intra_entropy shape={per_res_intra_entropy.shape}, values={per_res_intra_entropy[:5]}")

        # l=binder_length aggregation: mean over all binder residues
        intra_loss = jnp.mean(per_res_intra_entropy)
        logger.debug(f"intra_loss unweighted={intra_loss}")
        intra_contrib = w_intra * intra_loss
        total_loss += intra_contrib
        logger.debug(f"weighted intra contribution = {w_intra} * {intra_loss} = {intra_contrib}")
        loss_breakdown['contact_intra'] = intra_loss
    else:
        loss_breakdown['contact_intra'] = 0.0

    w_inter = weights['boltz_contact_inter']
    logger.debug(f"w_inter={w_inter}")
    inter_loss_val = 0.0 # Initialize
    if w_inter > 0:
        # Inter-chain loss (binding interface)
        it_entropy = inter_pair_entropy[jnp.ix_(binder_indices, target_indices)] # (Lb, Lt)
        logger.debug(f"it_entropy shape={it_entropy.shape}")

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

        logger.debug(f"After bounds check k={k}, l={l}")

        # Calculate per-binder-residue best k contacts
        # Sort contacts for each binder residue: (Lb, Lt) -> (Lb, Lt)
        it_sorted = jnp.sort(it_entropy, axis=-1)
        logger.debug(f"it_sorted shape={it_sorted.shape}")

        # Select top k contacts: (Lb, Lt) -> (Lb, k)
        # Handle case where k > num_targets safely
        top_k_contacts = it_sorted[:, :k]
        logger.debug(f"top_k_contacts shape={top_k_contacts.shape}")

        # Average the top k contacts for each binder residue: (Lb, k) -> (Lb,)
        per_res_mean_top_k = jnp.mean(top_k_contacts, axis=-1)
        logger.debug(f"per_res_mean_top_k shape={per_res_mean_top_k.shape}, values={per_res_mean_top_k[:5]}")

        # Select the top l binder residues based on their mean top-k contact entropy
        # Sort binder residues by their score: (Lb,) -> (Lb,)
        binder_scores_sorted = jnp.sort(per_res_mean_top_k)
        logger.debug(f"binder_scores_sorted shape={binder_scores_sorted.shape}")

        # Take the mean of the best l binder residues
        # Handle case where l > num_binders safely
        inter_loss_val = jnp.mean(binder_scores_sorted[:l])
        logger.debug(f"inter_loss_val unweighted={inter_loss_val}")

        inter_contrib = w_inter * inter_loss_val
        total_loss += inter_contrib
        logger.debug(f"weighted inter contribution = {w_inter} * {inter_loss_val} = {inter_contrib}")
        loss_breakdown['contact_inter'] = inter_loss_val
    else:
        loss_breakdown['contact_inter'] = 0.0

    logger.debug(f"total_distogram_loss weighted={total_loss}, breakdown={loss_breakdown}")
    return total_loss, loss_breakdown

def calculate_boltzdesign_binder_loss(
    partial_result: Dict[str, Any], # Model output (potentially partial)
    feature_dict: Dict[str, Any],
    target_indices: jnp.ndarray,
    binder_indices: jnp.ndarray,
    seq_representation: jnp.ndarray, # Softmax probs or STE
    weights_config: LossWeightsConfig, # Use typed config
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
        weights_config: LossWeightsConfig object containing loss weights.
        compute_confidence_loss: If True, compute and add pLDDT/PAE losses.
                                   Defaults to True for backward compatibility.
        inter_k: Optional override for k in inter-chain distogram loss.
        inter_l: Optional override for l in inter-chain distogram loss.

    Returns:
        Tuple containing the total loss and a dictionary breakdown of loss components.
    """
    logger.debug("==== STARTING BOLTZDESIGN BINDER LOSS CALCULATION ====")
    logger.debug(
        f"calculate_boltzdesign_binder_loss called with compute_confidence_loss={compute_confidence_loss}, "
        f"inter_k={inter_k}, inter_l={inter_l}"
    )
    logger.debug(f"partial_result keys: {list(partial_result.keys())}")
    # Let's log 
    # Convert config to dict for easier weight lookup
    weights = dataclasses.asdict(weights_config)
    logger.debug(f"weights dict: {weights}")
    logger.debug(f"Compute confidence loss: {compute_confidence_loss}")

    total_loss = 0.0
    loss_breakdown = {}

    # --- Distogram Loss --- 
    # Requires distogram logits and bin edges
    if 'distogram' in partial_result and 'logits' in partial_result['distogram'] and 'bin_edges' in partial_result['distogram']:
        logger.debug("Calling calculate_boltzdesign_distogram_loss...")
        distogram_loss, disto_breakdown = calculate_boltzdesign_distogram_loss(
            distogram_logits=partial_result['distogram']['logits'],
            bin_edges=partial_result['distogram']['bin_edges'],
            binder_indices=binder_indices,
            target_indices=target_indices,
            weights=weights,
            inter_k=inter_k,
            inter_l=inter_l
        )
        logger.debug(f"[Loss Sign Check] Distogram loss (+minimize): intra={disto_breakdown.get('contact_intra', 0.0):.4f}, inter={disto_breakdown.get('contact_inter', 0.0):.4f}")
        total_loss += distogram_loss
        loss_breakdown.update(disto_breakdown)
    else:
        logger.warning("Missing distogram logits or bin_edges in partial_result. Skipping distogram loss.")
        loss_breakdown['contact_intra'] = 0.0
        loss_breakdown['contact_inter'] = 0.0

    # --- Confidence Losses (Conditional) --- 
    # Use weights from typed config
    w_confidence = weights_config.boltz_confidence
    logger.debug(f"w_confidence={w_confidence}")
    if compute_confidence_loss:
        if w_confidence > 0:
            conf_loss = 0.0
            # pLDDT Loss (on binder)
            if 'predicted_lddt' in partial_result:
                logger.debug(f"Calculating pLDDT loss...")
                plddt = partial_result['predicted_lddt'] # Shape (..., N)
                if plddt.ndim == 2:
                    plddt = plddt[0] # Remove potential batch dim
                binder_plddt = plddt[binder_indices]
                # Use negative pLDDT as loss (minimize -pLDDT -> maximize pLDDT)
                # Scale by confidence weight
                plddt_neg_mean_unweighted = -jnp.mean(binder_plddt)
                logger.debug(f"[Loss Sign Check] pLDDT component (-maximize): {plddt_neg_mean_unweighted:.4f}")
                conf_loss += plddt_neg_mean_unweighted
                loss_breakdown['plddt_neg_mean'] = plddt_neg_mean_unweighted # Store the value being minimized
            else:
                logger.warning("Missing 'predicted_lddt' in partial_result. Skipping pLDDT confidence loss.")
                loss_breakdown['plddt_neg_mean'] = 0.0

            # Interface PAE Loss 
            if 'full_pae' in partial_result:
                logger.debug("Calculating PAE loss...")
                pae = partial_result['full_pae'] # Shape (..., N, N)
                if pae.ndim == 3:
                    pae = pae[0]
                interface_pae = pae[jnp.ix_(binder_indices, target_indices)]
                # Use mean interface PAE as loss (minimize PAE)
                pae_loss_unweighted = jnp.mean(interface_pae)
                logger.debug(f"[Loss Sign Check] PAE interface component (+minimize): {pae_loss_unweighted:.4f}")
                conf_loss += pae_loss_unweighted
                loss_breakdown['pae_inter_mean'] = pae_loss_unweighted # Store the value being minimized
            else:
                 logger.warning("Missing 'full_pae' in partial_result. Skipping PAE confidence loss.")
                 loss_breakdown['pae_inter_mean'] = 0.0

            # Add weighted confidence loss to total
            total_loss += w_confidence * conf_loss
            logger.debug(f"total confidence weighted contribution={w_confidence * conf_loss}")
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

    # --- Helix Loss ---
    w_helix = weights_config.boltz_helix
    if w_helix > 0:
        # Calculate helix loss if distogram data is available
        disto = partial_result.get('distogram', {})
        logits = disto.get('logits')
        bin_edges = disto.get('bin_edges')
        if logits is not None and bin_edges is not None:
            num_residues = logits.shape[-2]
            residue_index = jnp.arange(num_residues)
            helix_loss_unweighted = calculate_helix_loss(logits, bin_edges, residue_index, binder_indices)
            logger.debug(f"[Loss Sign Check] Helix component (+minimize): {helix_loss_unweighted:.4f}")
        else:
            helix_loss_unweighted = jnp.array(0.0, dtype=jnp.float32)
            logger.debug("Distogram logits or bin_edges missing for helix loss, setting helix_loss to 0")
        total_loss += w_helix * helix_loss_unweighted
        loss_breakdown['helix'] = helix_loss_unweighted
    else:
        loss_breakdown['helix'] = 0.0

    logger.debug("Completed BoltzDesign loss calculation")
    logger.debug(f"Total Loss: {total_loss}, Breakdown: {loss_breakdown}")
    logger.debug(f"calculate_boltzdesign_binder_loss final total_loss={total_loss}, breakdown={loss_breakdown}")
    return total_loss, loss_breakdown 
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
import functools # Added for partial

# Assume these loss calculation helper functions are correct as provided earlier
# If they also contain Python `if` statements dependent on JAX tracers,
# they might need similar refactoring using jax.lax.cond or other JAX primitives.

# --- Helper for Distogram Entropy Loss (Paper Eq 2-5) ---
def entropy_low_bins(dgram_logits, low_mask):
    """Calculates entropy over bins masked by low_mask. JAX-friendly.

    Args:
        dgram_logits: Raw distogram logits (..., num_bins).
        low_mask: Boolean mask for low-distance bins (num_bins,).

    Returns:
        Entropy value(s) (...).
    """
    # Ensure inputs are JAX arrays
    dgram_logits = jnp.asarray(dgram_logits)
    low_mask = jnp.asarray(low_mask, dtype=jnp.float32) # Use float for masking trick

    q = jax.nn.softmax(dgram_logits, axis=-1)          # Eq.3   shape (..., 64)
    # Mask high-distance bins using large negative offset before softmax (Eq.5 trick)
    q_star_logits = dgram_logits + (-1e7 * (1.0 - low_mask))
    q_star = jax.nn.softmax(q_star_logits, axis=-1) # Probabilities only over low bins

    # Calculate entropy using original probabilities 'q' but summed over q_star support (Eq.2)
    # Add epsilon for numerical stability in log
    entropy = -jnp.sum(q_star * jnp.log(jnp.maximum(q, 1e-9)), axis=-1)
    return entropy # Shape (...) e.g., (R, R) if input was (R, R, 64)
# --- End Helper ---


def get_binder_plddt_loss(result, binder_indices):
    """Calculate loss based on pLDDT values for binder residues. (Simplified Shape Handling)

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
    binder_indices = jnp.asarray(binder_indices) # Ensure indices are JAX array

    # --- Simplified Shape Handling: Assume Rank 3 (samples, residues, atoms) ---
    # If plddt might sometimes be rank 2 (residues, atoms), add a check or handle appropriately.
    # For now, proceed assuming rank 3 based on original logic.
    try:
        # Correct indexing: axis 1 for residues
        # binder_indices should be valid for the padded dimension
        binder_plddt_samples_atoms = plddt[:, binder_indices, :] # Assumes rank 3

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

def get_interface_pae_loss(result, target_indices, binder_indices):
    """Calculate loss based on PAE values at the interface.

    Args:
        result: AlphaFold model result dictionary or confidence head output.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.

    Returns:
        Average PAE at the interface (target-binder interactions).
    """
    # Handle different output formats
    pae = None
    if isinstance(result, dict):
        # Check various possible PAE keys
        for pae_key in ['full_pae', 'pae']:  # Prioritize 'full_pae' if available
            if pae_key in result:
                pae = result[pae_key]
                break
        else:
            logging.warning("Could not find PAE data in result. Format may be incorrect.")
            return jnp.array(0.0, dtype=jnp.float32)  # Return JAX array
    else:
        logging.warning("Result is not a dictionary. Cannot extract PAE.")
        return jnp.array(0.0, dtype=jnp.float32)  # Return JAX array

    pae = jnp.asarray(pae)  # Ensure it's a JAX array
    # jax.debug.print(">>> get_interface_pae_loss: Input pae shape={s}, rank={r}", 
    #                 s=pae.shape, r=pae.ndim)

    # --- JAX-friendly shape handling for potential batch dimension ---
    pae_rank = pae.ndim

    def process_pae_rank_gt_2(arr):
        # Average over the first (batch/sample) dimension
        processed_arr = jnp.mean(arr, axis=0)  # Output shape is (R, R)
        # jax.debug.print(">>> process_pae_rank_gt_2: Input shape={s}, Output shape={o}",
        #                 s=arr.shape, o=processed_arr.shape)
        return processed_arr

    def process_pae_rank_le_2(arr):
        # For rank <= 2, unify shape to (R, R) by ignoring any leading dimension of size 1
        # jax.debug.print(">>> process_pae_rank_le_2: Input arr shape={s}, rank={r}", 
        #                 s=arr.shape, r=arr.ndim)
        # Reshape to the final two dimensions (R, R)
        processed_le_2 = jnp.reshape(arr, arr.shape[-2:])
        # jax.debug.print(">>> process_pae_rank_le_2: Output shape={s}", 
        #                 s=processed_le_2.shape)
        return processed_le_2

    # Use a cond to handle whether there's a batch dimension
    pae_processed = jax.lax.cond(
        jnp.greater(pae_rank, 2),  # If rank > 2, assume shape (S, R, R)
        process_pae_rank_gt_2,     # -> reduce over first dim to get (R, R)
        process_pae_rank_le_2,     # -> either identity if it's (R, R) or squeeze if (1, R, R)
        pae
    )
    # jax.debug.print(">>> get_interface_pae_loss: pae_processed shape={s}", 
    #                 s=pae_processed.shape)
    # Now pae_processed should always be (R, R).

    # Ensure indices are JAX arrays
    target_indices_jax = jnp.asarray(target_indices)
    binder_indices_jax = jnp.asarray(binder_indices)

    # --- JAX-friendly check for 2D shape before indexing ---
    def index_pae_2d(arr_2d):
        # jax.debug.print(">>> index_pae_2d: Input shape={s}", s=arr_2d.shape)
        interface_pae = arr_2d[jnp.ix_(target_indices_jax, binder_indices_jax)]
        # Use nan_to_num before mean for safety
        return jnp.mean(jnp.nan_to_num(interface_pae))

    def handle_non_2d_pae(arr_non_2d):
        # jax.debug.print(">>> handle_non_2d_pae: Input shape={s}", s=arr_non_2d.shape)
        return jnp.array(0.0, dtype=jnp.float32)

    pae_loss = jax.lax.cond(
        jnp.equal(pae_processed.ndim, 2),
        index_pae_2d,
        handle_non_2d_pae,
        pae_processed
    )
    # jax.debug.print(">>> get_interface_pae_loss: Final pae_loss={p}", p=pae_loss)

    return pae_loss


# --- REMOVED OLD FUNCTION ---
# def get_contact_loss(result, target_indices, binder_indices):
#     """Calculate loss based on contact probability. (DEPRECATED)"""
#     # ... old implementation based on contact_probs ...
#     pass
# --- END REMOVED OLD FUNCTION ---


def get_target_fape_loss(result, initial_coords, target_indices):
    """Calculate loss to maintain target structure using FAPE-like RMSD.

    Args:
        result: AlphaFold model result dictionary.
        initial_coords: Initial coordinates of the structure (NumPy or JAX array).
        target_indices: Indices of the target residues (NumPy or JAX array).

    Returns:
        FAPE-like loss for target residues.
    """
    # Get final coordinates from result
    if 'final_atom_positions' not in result:
        logging.warning("final_atom_positions not found in result for FAPE calculation.")
        return jnp.array(0.0, dtype=jnp.float32)

    final_coords = jnp.asarray(result['final_atom_positions'])
    initial_coords = jnp.asarray(initial_coords) # Ensure JAX array
    target_indices = jnp.asarray(target_indices) # Ensure JAX array

    # Add checks for compatible shapes before indexing - These Python checks might fail under JIT/grad
    # It's safer to perform shape checks outside the JITted function or handle potential errors inside.
    # For now, assume shapes are compatible based on typical usage.

    try:
        # Select coordinates for target residues
        # Assuming coordinates shape is (..., num_residues_padded, num_atoms, 3)
        target_initial_coords = initial_coords[..., target_indices, :, :]
        target_final_coords = final_coords[..., target_indices, :, :]

        # Create mask for valid atoms (non-zero coordinates)
        valid_mask_initial = jnp.any(target_initial_coords != 0, axis=-1)
        valid_mask_final = jnp.any(target_final_coords != 0, axis=-1)
        valid_mask = valid_mask_initial & valid_mask_final

        # Calculate squared distances, applying the mask
        squared_diff = jnp.sum(
            jnp.square(target_final_coords - target_initial_coords) * valid_mask[..., None],
            axis=-1 # Sum over xyz coordinates
        )

        # Calculate mean squared error, considering only valid atoms
        sum_sq_diff = jnp.sum(squared_diff)
        num_valid_atoms = jnp.sum(valid_mask)
        safe_denom = jnp.maximum(num_valid_atoms, 1e-8)
        safe_rmsd_like_loss = sum_sq_diff / safe_denom

        # Use nan_to_num for final safety
        return jnp.nan_to_num(safe_rmsd_like_loss)

    except Exception as e:
        # Catch potential shape or indexing errors during JAX operations
        logging.error(f"Error during FAPE calculation (shapes final:{final_coords.shape}, initial:{initial_coords.shape}, indices:{target_indices.shape}): {e}")
        return jnp.array(0.0, dtype=jnp.float32)


def get_binder_seq_entropy_loss(binder_seq_logits):
    """Calculate sequence entropy loss to encourage diversity.

    Args:
        binder_seq_logits: Sequence logits for the binder (JAX array).

    Returns:
        Negative mean sequence entropy (scalar JAX array).
    """
    # Ensure input is JAX array
    binder_seq_logits = jnp.asarray(binder_seq_logits)

    probs = jax.nn.softmax(binder_seq_logits, axis=-1)
    # Add epsilon for numerical stability before log
    log_probs = jnp.log(jnp.maximum(probs, 1e-8)) # Use maximum instead of add
    entropy = -jnp.sum(probs * log_probs, axis=-1) # Entropy per position

    # Return negative mean entropy (as we want to maximize entropy for diversity)
    # Use nan_to_num for safety before final mean
    return -jnp.mean(jnp.nan_to_num(entropy))


# --- REPLACED FUNCTION ---
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

        # Handle cases where num_binder < 2 or rows have fewer than 2 valid entries
        k_intra = 2
        def pick_k_or_less(row):
            # --- JIT-compatible way to get mean of k smallest finite values ---
            # Take the first k elements (guaranteed to be the smallest, might contain inf)
            first_k = row[:k_intra] # Static slice is okay
            
            # Count how many are finite
            finite_mask_k = jnp.isfinite(first_k)
            num_finite_in_k = jnp.sum(finite_mask_k)
            
            # Sum only the finite values among the first k
            # Use where to turn non-finite (inf) to 0 before summing
            sum_finite_k = jnp.sum(jnp.where(finite_mask_k, first_k, 0.0))
            
            # Calculate mean, avoiding division by zero
            mean_val = jnp.where(num_finite_in_k > 0, sum_finite_k / num_finite_in_k, 0.0)
            return mean_val
            # --- End JIT-compatible fix ---

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
        # Handle case where num_target might be 0 by returning inf
        safe_min = lambda x: jnp.min(x) if x.shape[0] > 0 else jnp.inf
        per_res_inter_min = jax.vmap(safe_min)(it_entropy) # Shape (num_binder,)

        # Final inter-loss: mean over all binder residues
        # Use nan_to_num, handle inf values by converting them to a large number or filtering
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
# --- END REPLACED FUNCTION ---


def calculate_gradient_binder_loss(result, feature_dict, target_indices, binder_indices, binder_seq_logits, design_params):
    """Calculate loss for gradient-based binder design. (Refactored for JAX control flow & New Distogram Loss)

    Args:
        result: AlphaFold model result dictionary (MUST contain 'distogram' with 'logits' and 'bin_edges').
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
        lambda op: get_binder_plddt_loss(op[0], op[1]), # Use lambda with operand tuple
        lambda op: zero_loss_float32,
        (result, binder_indices_jax) # Pass operands as tuple
    )

    # --- Calculate Interface PAE Loss ---
    pae_inter_weight = weights.get('pae_inter', 0.0)
    losses['pae_inter'] = jax.lax.cond(
        jnp.greater(jnp.asarray(pae_inter_weight), 0.0),
        lambda op: get_interface_pae_loss(op[0], op[1], op[2]),
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
            total_dist_loss, breakdown = get_distogram_entropy_loss(
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
            logging.warning("Distogram logits or bin_edges missing, cannot calculate entropy loss.")
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
        lambda logits: get_binder_seq_entropy_loss(logits),
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
             return get_target_fape_loss(res, feat_dict['initial_coords'], tgt_idx)
        else:
             # Avoid logging inside cond branches if possible
             # logging.warning("FAPE target weight > 0 but 'initial_coords' not in feature_dict.")
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
    loss_breakdown_unweighted = {}
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
    loss_breakdown_unweighted['contact_intra'] = losses['contact_intra']
    loss_breakdown_unweighted['contact_inter'] = losses['contact_inter']

    return total_loss, loss_breakdown_unweighted


def calculate_boltz_binder_loss(partial_result, feature_dict, target_indices, binder_indices, binder_seq_logits, design_params):
    """Calculate loss for BoltzDesign1-like binder design. (Refactored for JAX & New Distogram Loss)

    Args:
        partial_result: Partial model result (MUST contain 'distogram' with 'logits' and 'bin_edges').
        feature_dict: Feature dictionary.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.
        binder_seq_logits: Sequence logits for the binder.
        design_params: Dictionary of design parameters (expected to be the weights dict).

    Returns:
        Tuple of (total_loss, loss_breakdown).
    """
    # --- Validate inputs ---
    logging.info("==== STARTING BOLTZ BINDER LOSS CALCULATION (Refactored + Distogram Entropy) ====")
    # Avoid logging shapes of potential tracers directly with f-strings inside transformed functions
    # logging.info(f"Target indices shape: {target_indices.shape}, Binder indices shape: {binder_indices.shape}")
    # logging.info(f"Binder logits shape: {binder_seq_logits.shape}")

    # Perform Python-level checks if needed, before JAX operations
    # These checks might fail if run inside jit/grad on tracers. Best practice is to validate inputs outside.
    # Example (assuming indices are passed as NumPy arrays or lists initially):
    if isinstance(target_indices, (np.ndarray, list)) and len(target_indices) == 0:
         logging.error("Target indices array is empty! Cannot calculate loss.")
         raise ValueError("Empty target_indices array")
    if isinstance(binder_indices, (np.ndarray, list)) and len(binder_indices) == 0:
         logging.error("Binder indices array is empty! Cannot calculate loss.")
         raise ValueError("Empty binder_indices array")

    # Log dictionary keys (Python operations on dicts are usually fine)
    logging.info(f"partial_result keys: {list(partial_result.keys())}")
    logging.info(f"design_params (weights) keys: {list(design_params.keys())}")

    # design_params *is* the weights dictionary in this context
    weights = design_params # Assume design_params is already the weights dict
    logging.info(f"Actual Design weights passed: {weights}") # Standard logging

    # Initialize losses dictionary with new contact terms
    losses = {
        'contact_intra': jnp.array(0.0, dtype=jnp.float32),
        'contact_inter': jnp.array(0.0, dtype=jnp.float32),
        'plddt': jnp.array(0.0, dtype=jnp.float32),
        'pae': jnp.array(0.0, dtype=jnp.float32),
        'seq_entropy': jnp.array(0.0, dtype=jnp.float32),
        'seq_one_hot': jnp.array(0.0, dtype=jnp.float32) # Keep for consistency downstream
    }
    zero_loss_float32 = jnp.array(0.0, dtype=jnp.float32) # Reusable zero

    # Ensure indices are JAX arrays for calculations
    target_indices_jax = jnp.asarray(target_indices)
    binder_indices_jax = jnp.asarray(binder_indices)

    # --- Calculate Distogram Entropy Loss using jax.lax.cond ---
    w_intra = weights.get('contact_intra', 0.0)
    w_inter = weights.get('contact_inter', 0.0)
    distogram_pred = jnp.greater(jnp.asarray(w_intra) + jnp.asarray(w_inter), 0.0)

    def calc_distogram_true_boltz(operand): # Use operand
        pr, tgt_idx, bnd_idx, w = operand
        # This check can remain Pythonic as it's checking dictionary keys
        if 'distogram' in pr and isinstance(pr['distogram'], dict) and \
           'logits' in pr['distogram'] and 'bin_edges' in pr['distogram']:
            # Call the new loss function
            total_dist_loss, breakdown = get_distogram_entropy_loss(
                distogram_logits=pr['distogram']['logits'],
                bin_breaks=pr['distogram']['bin_edges'],
                target_indices=tgt_idx,
                binder_indices=bnd_idx,
                weights=w # Pass weights dict directly
            )
            # Return the breakdown (unweighted) and the total weighted loss
            return breakdown.get('contact_intra', zero_loss_float32), \
                   breakdown.get('contact_inter', zero_loss_float32), \
                   total_dist_loss
        else:
            logging.warning("Distogram logits or bin_edges missing for Boltz loss.")
            return zero_loss_float32, zero_loss_float32, zero_loss_float32

    def calc_distogram_false_boltz(_): # Operand ignored
        return zero_loss_float32, zero_loss_float32, zero_loss_float32

    # Calculate distogram losses conditionally
    loss_intra, loss_inter, weighted_distogram_loss = jax.lax.cond(
        distogram_pred,
        calc_distogram_true_boltz,
        calc_distogram_false_boltz,
        operand=(partial_result, target_indices_jax, binder_indices_jax, weights) # Pass necessary data
    )
    losses['contact_intra'] = loss_intra # Store unweighted
    losses['contact_inter'] = loss_inter # Store unweighted
    # Note: weighted_distogram_loss will be added to total_loss later

    # --- Calculate Confidence Losses (pLDDT, PAE) using jax.lax.cond ---
    confidence_weight = weights.get('confidence', 0.0)
    # Ensure weight is treated as a JAX value
    confidence_pred = jnp.greater(jnp.asarray(confidence_weight), 0.0)

    def calc_plddt_true(operand): # Use operand
        pr, bnd_idx = operand
        if 'predicted_lddt' in pr:
            plddt_loss_val = get_binder_plddt_loss(pr, bnd_idx)
            return jnp.nan_to_num(plddt_loss_val)
        else:
            return zero_loss_float32

    def calc_plddt_false(_): # Operand ignored
         return zero_loss_float32

    losses['plddt'] = jax.lax.cond(
        confidence_pred,
        calc_plddt_true,
        calc_plddt_false,
        operand=(partial_result, binder_indices_jax) # Pass necessary data
    )

    def calc_pae_true(operand): # Use operand
        pr, tgt_idx, bnd_idx = operand
        if 'full_pae' in pr:
             pae_loss_val = get_interface_pae_loss(pr, tgt_idx, bnd_idx)
             return jnp.nan_to_num(pae_loss_val)
        else:
             return zero_loss_float32

    def calc_pae_false(_): # Operand ignored
         return zero_loss_float32

    losses['pae'] = jax.lax.cond(
        confidence_pred, # Same predicate used for PAE
        calc_pae_true,
        calc_pae_false,
        operand=(partial_result, target_indices_jax, binder_indices_jax) # Pass necessary data
    )
    # --------------------------------------------------------------

    # --- Calculate Sequence Entropy Loss using jax.lax.cond ---
    seq_entropy_weight = weights.get('seq_entropy', 0.0)
    # Ensure weight is treated as a JAX value
    seq_entropy_pred = jnp.greater(jnp.asarray(seq_entropy_weight), 0.0)

    def calc_entropy_true(logits): # Pass logits as operand
        entropy_loss_val = get_binder_seq_entropy_loss(logits)
        return jnp.nan_to_num(entropy_loss_val)

    def calc_entropy_false(_): # Operand ignored
        return zero_loss_float32

    losses['seq_entropy'] = jax.lax.cond(
        seq_entropy_pred,
        calc_entropy_true,
        calc_entropy_false,
        operand=binder_seq_logits # Pass the necessary data
    )
    # -------------------------------------------------------

    # Calculate weighted total loss
    total_loss = zero_loss_float32
    loss_breakdown_unweighted = {} # Create a new dict for unweighted losses

    # Iterate through the keys of the initialized losses dict for consistency
    for k in losses:
        if k in ['seq_one_hot']: # Skip seq_one_hot handled elsewhere
            continue
        if k in ['contact_intra', 'contact_inter']:
            # These are handled via weighted_distogram_loss, store unweighted only
            loss_breakdown_unweighted[k] = losses[k]
            continue

        loss_val = losses[k] # Get calculated loss (already defaults to 0 if not calculated)

        # Handle combined confidence weight logic
        if k == 'plddt' or k == 'pae':
            # Use the 'confidence' weight if present, otherwise default to 0
            weight_val = weights.get('confidence', 0.0)
            # If you intend separate weights, check for weights.get('plddt') etc.
        else:
            # Get specific weight for other loss terms
            weight_val = weights.get(k, 0.0)

        # Ensure weight is a JAX array for multiplication
        weighted_loss = jnp.asarray(weight_val) * loss_val
        total_loss = total_loss + weighted_loss
        loss_breakdown_unweighted[k] = loss_val # Store the unweighted loss value

    # Add the pre-weighted distogram loss
    total_loss = total_loss + weighted_distogram_loss

    # Add the placeholder for seq_one_hot back for consistent structure if needed by caller
    loss_breakdown_unweighted['seq_one_hot'] = losses['seq_one_hot']
    # Ensure distogram losses are in the final breakdown
    loss_breakdown_unweighted['contact_intra'] = losses['contact_intra']
    loss_breakdown_unweighted['contact_inter'] = losses['contact_inter']


    logging.info("Completed loss calculation (Refactored + Distogram Entropy)")

    # Return the dictionary including potentially zeroed values for consistent structure
    return total_loss, loss_breakdown_unweighted


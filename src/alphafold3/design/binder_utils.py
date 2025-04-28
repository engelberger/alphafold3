import numpy as np
import jax
import jax.numpy as jnp
from absl import logging
from alphafold3.common import folding_input
from alphafold3.design.config import ALPHABET_SIZE # Import alphabet size
from typing import Dict, Any

def get_residue_indices(int_asym_ids, target_chains, binder_chains, int_to_str_chain_map):
    """Maps integer asym_ids to residue indices for target and binder.
    
    Args:
        int_asym_ids: Array of integer asym_ids for each residue.
        target_chains: List of *string* chain IDs that should remain fixed.
        binder_chains: List of *string* chain IDs that should be designed.
        int_to_str_chain_map: Dictionary mapping integer asym_ids to string chain IDs.
        
    Returns:
        A tuple of (target_indices, binder_indices) as numpy arrays.
    """
    target_idx = []
    binder_idx = []
    
    # Add detailed logging inside the loop
    logging.info(f"Starting residue index assignment. int_asym_ids length: {len(int_asym_ids)}")
    assigned_binder_count = 0
    assigned_target_count = 0
    
    for i, int_asym_id_val in enumerate(int_asym_ids):
        int_asym_id = int(int_asym_id_val) # Ensure it's a Python int for dict key
        # Convert integer ID to string ID using the map
        str_chain_id = int_to_str_chain_map.get(int_asym_id, None)
        
        assignment = "None"
        if str_chain_id is None:
            # Log skipped residues differently
            if i < 196: # Only log warnings for potentially real residues
                 logging.warning(f"Residue {i}: Int ID {int_asym_id} -> No string map. Skipping.")
            assignment = "Skipped (No Map)"
            # continue # Keep continue commented out for now to see all indices
        
        # Compare using string IDs
        elif str_chain_id in target_chains:
            target_idx.append(i)
            assigned_target_count += 1
            assignment = "Target"
        elif str_chain_id in binder_chains:
            binder_idx.append(i)
            assigned_binder_count += 1
            assignment = "Binder"
        else:
            # Log residues that map but don't match target/binder
            if i < 196:
                logging.warning(f"Residue {i}: Int ID {int_asym_id} -> Str ID '{str_chain_id}'. Not target {target_chains} or binder {binder_chains}. Skipping.")
            assignment = f"Skipped (Mapped to '{str_chain_id}')"

        # Log details for specific ranges or all if needed
        if i < 5 or (160 <= i < 200) or i >= 250:
             logging.info(f"  Index {i}: Int ID {int_asym_id} -> Str ID '{str_chain_id}' -> Assigned: {assignment}")

    logging.info(f"Finished assignment loop. Assigned {assigned_target_count} to target, {assigned_binder_count} to binder.")

    if not target_idx:
        # Provide more context in error message
        logging.error(f"Map used for conversion: {int_to_str_chain_map}")
        logging.error(f"Integer asym_ids found in features: {np.unique(int_asym_ids)}")
        raise ValueError(f"Could not find any residues for target chains {target_chains} after mapping.")
    if not binder_idx:
        logging.error(f"Map used for conversion: {int_to_str_chain_map}")
        logging.error(f"Integer asym_ids found in features: {np.unique(int_asym_ids)}")
        raise ValueError(f"Could not find any residues for binder chains {binder_chains} after mapping.")
    
    logging.info(f"Target indices ({len(target_idx)}): {target_idx[:5]}...{target_idx[-5:] if len(target_idx) > 5 else target_idx}")
    logging.info(f"Binder indices ({len(binder_idx)}): {binder_idx[:5]}...{binder_idx[-5:] if len(binder_idx) > 5 else binder_idx}")
    
    return np.array(target_idx), np.array(binder_idx)

def setup_binder_features(feature_dict, target_chains, binder_chains, fold_input: folding_input.Input):
    """Modifies feature_dict for binder design.
    
    Args:
        feature_dict: Dictionary of model features.
        target_chains: List of chain IDs that should remain fixed (target).
        binder_chains: List of chain IDs that should be designed (binder).
        fold_input: The original folding_input.Input object.
        
    Returns:
        Tuple of (modified_feature_dict, target_indices, binder_indices)
    """
    logging.info("Setting up features for binder design...")
    # --- Log Feature Dict Keys ---
    logging.info(f"Available feature_dict keys: {list(feature_dict.keys())}")
    
    # --- STEP 1: Log chain details from fold_input for debugging ---
    total_expected_length = 0
    chain_details = []
    
    for i, chain in enumerate(fold_input.chains):
        chain_id_str = chain.id
        if isinstance(chain, folding_input.ProteinChain):
            chain_len = len(chain.sequence)
            chain_details.append(f"Chain {i} (ID: {chain_id_str}): Protein, Length={chain_len}, First 10 aa: {chain.sequence[:10]}...")
        elif isinstance(chain, folding_input.RnaChain):
            chain_len = len(chain.sequence)
            chain_details.append(f"Chain {i} (ID: {chain_id_str}): RNA, Length={chain_len}")
        elif isinstance(chain, folding_input.DnaChain):
            chain_len = len(chain.sequence)
            chain_details.append(f"Chain {i} (ID: {chain_id_str}): DNA, Length={chain_len}")
        elif isinstance(chain, folding_input.Ligand):
            chain_len = len(chain.ccd_ids) if chain.ccd_ids else (1 if chain.smiles else 0)
            chain_details.append(f"Chain {i} (ID: {chain_id_str}): Ligand, Length={chain_len}")
        else:
            chain_details.append(f"Chain {i} (ID: {chain_id_str}): Unknown type={type(chain)}")
            chain_len = 0
        
        total_expected_length += chain_len
    
    # Log all chain details
    logging.info(f"========== FOLD INPUT CHAIN DETAILS ==========")
    for detail in chain_details:
        logging.info(detail)
    logging.info(f"Total expected residue length: {total_expected_length}")
    
    # --- STEP 2: Get Chain ID to Residue Mapping ---
    # Check available chain ID keys
    chain_id_key = None
    for possible_key in ['token_features.asym_id', 'asym_id', 'entity_id']:
        if possible_key in feature_dict:
            chain_id_key = possible_key
            break
    
    if not chain_id_key:
        logging.error("Available feature dict keys: " + str(list(feature_dict.keys())))
        raise KeyError("No chain ID key found in feature_dict.")
    
    int_asym_ids = feature_dict[chain_id_key]
    logging.info(f"Using {chain_id_key} for chain mapping with shape {int_asym_ids.shape}")
    logging.info(f"Unique values in {chain_id_key}: {np.unique(int_asym_ids)}")
    
    # --- STEP 3: Calculate target and binder indices properly ---
    # Get residue indices based on actual sequence in each chain
    target_idx = []
    binder_idx = []
    current_idx = 0
    
    for i, chain in enumerate(fold_input.chains):
        chain_id_str = chain.id
        
        # Calculate chain length
        if isinstance(chain, (folding_input.ProteinChain, folding_input.RnaChain, folding_input.DnaChain)):
            chain_len = len(chain.sequence)
        elif isinstance(chain, folding_input.Ligand):
            chain_len = len(chain.ccd_ids) if chain.ccd_ids else (1 if chain.smiles else 0)
        else:
            logging.warning(f"Skipping unknown chain type: {type(chain)}")
            continue
            
        # Skip empty chains
        if chain_len == 0:
            logging.warning(f"Skipping empty chain {chain_id_str}")
            continue
            
        # Calculate end index (exclusive)
        end_idx = current_idx + chain_len
        
        # Make sure we don't exceed max sequence length
        max_seq_len = min(len(int_asym_ids), 1000)  # Safety limit
        if current_idx >= max_seq_len:
            logging.warning(f"Chain {chain_id_str} starts beyond max sequence length ({current_idx} >= {max_seq_len})")
            continue
            
        # Calculate actual indices for this chain (clamp to valid range)
        chain_end = min(end_idx, max_seq_len)
        chain_indices = list(range(current_idx, chain_end))
        
        # Detailed logging
        if len(chain_indices) > 0:
            logging.info(f"Chain {chain_id_str}: Assigned indices {chain_indices[0]}...{chain_indices[-1]} ({len(chain_indices)} residues)")
            
            # Assign to target or binder
            if chain_id_str in target_chains:
                target_idx.extend(chain_indices)
                logging.info(f"  -> TARGET CHAIN: Added {len(chain_indices)} indices to target set")
            elif chain_id_str in binder_chains:
                binder_idx.extend(chain_indices)
                logging.info(f"  -> BINDER CHAIN: Added {len(chain_indices)} indices to binder set")
            else:
                logging.info(f"  -> IGNORED CHAIN: Not target or binder")
        else:
            logging.warning(f"Chain {chain_id_str}: No valid indices in range [{current_idx}:{chain_end}]")
        
        # Update current index for next chain
        current_idx = end_idx
    
    target_indices = np.array(target_idx)
    binder_indices = np.array(binder_idx)
    
    # --- STEP 4: Validate results ---
    if len(target_indices) == 0:
        raise ValueError(f"Found 0 target indices for chains {target_chains}. Check fold_input and feature_dict.")
    
    if len(binder_indices) == 0:
        raise ValueError(f"Found 0 binder indices for chains {binder_chains}. Check fold_input and feature_dict.")
    
    logging.info(f"========== FINAL INDICES ==========")
    logging.info(f"Target indices ({len(target_indices)}): {target_indices[:5]}...{target_indices[-5:] if len(target_indices) > 5 else target_indices}")
    logging.info(f"Binder indices ({len(binder_indices)}): {binder_indices[:5]}...{binder_indices[-5:] if len(binder_indices) > 5 else binder_indices}")
    
    # --- STEP 5: Set up design mask and other features ---
    # Check for mask key
    seq_mask_key = None
    for possible_key in ['seq_mask', 'token_features.mask', 'sequence_mask']:
        if possible_key in feature_dict:
            seq_mask_key = possible_key
            break
    
    if not seq_mask_key:
        logging.error("Available feature dict keys: " + str(list(feature_dict.keys())))
        raise KeyError("No sequence mask key found in feature_dict.")
    
    logging.info(f"Using key '{seq_mask_key}' for design mask")
    
    # Create design mask
    design_mask = np.zeros_like(feature_dict[seq_mask_key])
    design_mask[binder_indices] = 1
    feature_dict['design_mask'] = design_mask
    logging.info(f"Added 'design_mask' with {int(design_mask.sum())} designable positions")
    
    # Store initial coordinates (for target FAPE loss)
    atom_pos_key = 'template_atom_positions'
    if atom_pos_key in feature_dict:
        initial_coords = feature_dict[atom_pos_key].copy()
        feature_dict['initial_coords'] = initial_coords
        logging.info(f"Stored 'initial_coords' (from {atom_pos_key}) for potential target FAPE loss")
    else:
        logging.warning(f"Could not find '{atom_pos_key}' to store initial coordinates")
    
    # Mask binder MSA
    msa_key = 'msa'
    msa_mask_key = 'msa_mask'
    if msa_key in feature_dict and msa_mask_key in feature_dict:
        logging.info(f"Masking MSA features (key: '{msa_key}') for binder residues")
        try:
            # Log MSA shape
            logging.info(f"Shape of feature_dict['{msa_key}']: {feature_dict[msa_key].shape}")
            logging.info(f"Shape of feature_dict['{msa_mask_key}']: {feature_dict[msa_mask_key].shape}")
            
            # Ensure binder_indices are valid for the sequence length dimension
            seq_len = feature_dict[msa_key].shape[1]
            valid_binder_indices = binder_indices[binder_indices < seq_len]
            
            if len(valid_binder_indices) > 0:
                # Mask MSA for binder
                feature_dict[msa_key][:, valid_binder_indices] = 0
                feature_dict[msa_mask_key][:, valid_binder_indices] = 0
                logging.info(f"Masked {len(valid_binder_indices)} positions in MSA for binder residues")
            else:
                logging.warning("No valid binder indices found within MSA sequence length")
        except IndexError as e:
            logging.error(f"Error masking binder MSA: {e}")
            raise
    else:
        logging.warning(f"Could not find '{msa_key}' or '{msa_mask_key}' to mask binder MSA")
    
    logging.info("Binder feature setup complete")
    return feature_dict, target_indices, binder_indices

def update_features_from_logits(
    feature_dict: Dict[str, Any],
    binder_indices: jnp.ndarray,
    seq_repr_dict: Dict[str, jnp.ndarray] # New input from soft_seq_af3
) -> Dict[str, Any]:
    """Updates feature_dict based on sequence representations (including STE).

    Args:
        feature_dict: Dictionary of model features.
        binder_indices: Indices of the binder residues.
        seq_repr_dict: Dictionary containing sequence representations
                       (output of sequence_utils.soft_seq_af3), expected keys:
                       'hard' (STE one-hot), 'pseudo' (differentiable mixture).

    Returns:
        Updated feature dictionary.
    """
    from absl import logging

    # Ensure binder_indices is JAX array
    binder_indices = jnp.asarray(binder_indices)

    # Extract needed representations
    if 'hard' not in seq_repr_dict or 'pseudo' not in seq_repr_dict:
        raise ValueError("seq_repr_dict must contain 'hard' and 'pseudo' keys.")
    ste_hard = seq_repr_dict['hard'] # Shape (L_binder, ALPHABET_SIZE)
    pseudo_probs = seq_repr_dict['pseudo'] # Shape (L_binder, ALPHABET_SIZE)

    # --- Determine discrete aatype from STE hard representation --- #
    # This provides the discrete sequence info needed by parts of the model
    binder_aatype_ste = jnp.argmax(ste_hard, axis=-1) # Shape (L_binder,)
    logging.debug(f"binder_aatype_ste shape: {binder_aatype_ste.shape}")

    # --- Create differentiable MSA profile/first row from pseudo probs --- #
    # This is used for features requiring differentiable sequence input
    # For MSA profile, it's directly the probabilities
    binder_profile = pseudo_probs # Shape (L_binder, ALPHABET_SIZE)
    # For the first row of MSA (often used as target sequence representation),
    # we might need the one-hot encoding *of the pseudo probabilities*
    # Let's use the one-hot derived from the STE for consistency in discrete parts
    # and use the pseudo_probs for continuous parts like profile.
    binder_msa_first_row_ste = ste_hard # Shape (L_binder, ALPHABET_SIZE)

    logging.debug(f"binder_profile (from pseudo) shape: {binder_profile.shape}")
    logging.debug(f"binder_msa_first_row_ste (from hard) shape: {binder_msa_first_row_ste.shape}")

    if 'design_mask' not in feature_dict:
        logging.warning("'design_mask' not found in feature_dict. Adding default mask based on binder_indices.")
        # Determine sequence length from a reliable feature like seq_mask or token_features.mask
        seq_len_key = next((k for k in ['seq_mask', 'token_features.mask'] if k in feature_dict), None)
        if seq_len_key:
             seq_len = feature_dict[seq_len_key].shape[0]
        else:
             # Fallback: try inferring from aatype if present
             seq_len = feature_dict.get('aatype', jnp.zeros((100,))).shape[0]
             logging.warning(f"Could not find seq_mask, inferring seq_len={seq_len} from aatype.")

        design_mask = jnp.zeros((seq_len,), dtype=jnp.int32)
        design_mask = design_mask.at[binder_indices].set(1)
        feature_dict['design_mask'] = design_mask

    # --- Update Feature Dictionary --- #
    # Make a copy to avoid in-place mutation issues with JAX
    updated_feature_dict = {**feature_dict} # Shallow copy is usually sufficient for JAX

    # Process each key in the feature dict
    for k, v in feature_dict.items():
        if k == 'aatype':
            # Update 'aatype' using the discrete STE-derived type
            target_dtype = v.dtype
            # Use functional scatter update for JAX compatibility
            updated_feature_dict[k] = v.at[binder_indices].set(binder_aatype_ste.astype(target_dtype))
            logging.debug(f"Updated {k} using STE argmax with shape {updated_feature_dict[k].shape}")

        elif k == 'msa' and isinstance(v, (np.ndarray, jnp.ndarray)) and v.ndim >= 2 and v.shape[0] > 0:
            # Update the first row of MSA (often target sequence) using STE one-hot
            # Check if v is a dict (as seen in logs) or array
            # Based on logs, feature_dict['msa'] can be a dict {'rows': ..., 'profile': ...}
            # We need to handle this structure.
            if isinstance(v, dict):
                current_msa_rows = v.get('rows')
                current_msa_profile = v.get('profile')
                updated_msa_dict = {**v} # Copy the dict

                if current_msa_rows is not None and current_msa_rows.ndim >= 2 and current_msa_rows.shape[0] > 0:
                    target_dtype_rows = current_msa_rows.dtype
                    # Update first row with STE hard representation (one-hot)
                    # Need to ensure binder_msa_first_row_ste has correct shape (L_binder, num_residue_types+1?)
                    # If MSA rows are integer indices, convert STE one-hot back to indices
                    if jnp.issubdtype(target_dtype_rows, jnp.integer):
                        msa_row_indices = jnp.argmax(binder_msa_first_row_ste, axis=-1)
                        logging.debug(f"Updating MSA rows[0] (int) at binder indices.")
                        updated_msa_dict['rows'] = current_msa_rows.at[0, binder_indices].set(msa_row_indices.astype(target_dtype_rows))
                    # If MSA rows are one-hot floats, use STE one-hot directly
                    elif jnp.issubdtype(target_dtype_rows, jnp.floating):
                         # Check if ALPHABET_SIZE matches the feature dimension
                         if binder_msa_first_row_ste.shape[-1] != current_msa_rows.shape[-1]:
                              logging.warning(f"MSA row feature dimension mismatch: STE ({binder_msa_first_row_ste.shape[-1]}) vs MSA ({current_msa_rows.shape[-1]}). Attempting zero-padding.")
                              # Pad STE representation if needed (e.g., for GAP/UNK)
                              padding = [(0,0)] * (binder_msa_first_row_ste.ndim - 1) + [(0, current_msa_rows.shape[-1] - binder_msa_first_row_ste.shape[-1])]
                              padded_ste = jnp.pad(binder_msa_first_row_ste, padding)
                         else:
                              padded_ste = binder_msa_first_row_ste
                         logging.debug(f"Updating MSA rows[0] (float) at binder indices.")
                         updated_msa_dict['rows'] = current_msa_rows.at[0, binder_indices].set(padded_ste.astype(target_dtype_rows))
                    else:
                         logging.warning(f"MSA rows have unexpected dtype {target_dtype_rows}. Skipping update.")

                # Update profile using the differentiable pseudo probabilities
                if current_msa_profile is not None:
                    target_dtype_profile = current_msa_profile.dtype
                     # Check if ALPHABET_SIZE matches the feature dimension
                    if binder_profile.shape[-1] != current_msa_profile.shape[-1]:
                        logging.warning(f"MSA profile feature dimension mismatch: Pseudo ({binder_profile.shape[-1]}) vs Profile ({current_msa_profile.shape[-1]}). Attempting zero-padding.")
                        # Pad pseudo probabilities if needed
                        padding = [(0,0)] * (binder_profile.ndim - 1) + [(0, current_msa_profile.shape[-1] - binder_profile.shape[-1])]
                        padded_profile = jnp.pad(binder_profile, padding)
                    else:
                        padded_profile = binder_profile
                    logging.debug(f"Updating MSA profile at binder indices.")
                    updated_msa_dict['profile'] = current_msa_profile.at[binder_indices].set(padded_profile.astype(target_dtype_profile))

                updated_feature_dict[k] = updated_msa_dict
                logging.debug(f"Updated MSA dict components.")
            else:
                 # Handle case where feature_dict['msa'] is an array (MSA rows or one-hot floats)
                 target_dtype = v.dtype
                 # Case 1: 2D array (MSA rows, possibly converted to float)
                 if v.ndim == 2:
                     # Update MSA rows: assign discrete aatype values to first row
                     msa_row_indices = jnp.argmax(binder_msa_first_row_ste, axis=-1)
                     updated_feature_dict[k] = v.at[0, binder_indices].set(msa_row_indices.astype(target_dtype))
                     logging.debug(f"Updated {k}[0] using STE argmax for 2D MSA rows with shape {updated_feature_dict[k].shape}")
                 # Case 2: integer dtype and >=2D (redundant but kept for clarity)
                 elif jnp.issubdtype(target_dtype, jnp.integer):
                     msa_row_indices = jnp.argmax(binder_msa_first_row_ste, axis=-1)
                     updated_feature_dict[k] = v.at[0, binder_indices].set(msa_row_indices.astype(target_dtype))
                     logging.debug(f"Updated {k}[0] (int) using STE argmax with shape {updated_feature_dict[k].shape}")
                 # Case 3: float dtype and 3D array (one-hot features)
                 elif jnp.issubdtype(target_dtype, jnp.floating) and v.ndim >= 3:
                     # One-hot float array: update using STE one-hot vectors
                     if binder_msa_first_row_ste.shape[-1] != v.shape[-1]:
                         logging.warning(f"MSA row feature dimension mismatch: STE ({binder_msa_first_row_ste.shape[-1]}) vs MSA ({v.shape[-1]}). Attempting zero-padding.")
                         padding = [(0,0)] * (binder_msa_first_row_ste.ndim - 1) + [(0, v.shape[-1] - binder_msa_first_row_ste.shape[-1])]
                         padded_ste = jnp.pad(binder_msa_first_row_ste, padding)
                     else:
                         padded_ste = binder_msa_first_row_ste
                     updated_feature_dict[k] = v.at[0, binder_indices].set(padded_ste.astype(target_dtype))
                     logging.debug(f"Updated {k}[0] (float) using STE one-hot with shape {updated_feature_dict[k].shape}")
                 else:
                     logging.warning(f"MSA array has unexpected shape/dtype (ndim={v.ndim}, dtype={target_dtype}). Skipping update.")
            
        elif k == 'msa_profile': # Explicitly handle msa_profile if it's separate
             target_dtype_profile = v.dtype
             if binder_profile.shape[-1] != v.shape[-1]:
                 logging.warning(f"MSA profile feature dimension mismatch: Pseudo ({binder_profile.shape[-1]}) vs Profile ({v.shape[-1]}). Attempting zero-padding.")
                 padding = [(0,0)] * (binder_profile.ndim - 1) + [(0, v.shape[-1] - binder_profile.shape[-1])]
                 padded_profile = jnp.pad(binder_profile, padding)
             else:
                 padded_profile = binder_profile
             updated_feature_dict[k] = v.at[binder_indices].set(padded_profile.astype(target_dtype_profile))
             logging.debug(f"Updated {k} using pseudo probs with shape {updated_feature_dict[k].shape}")

        # Note: Other features are implicitly copied by updated_feature_dict = {**feature_dict}

    # Check if essential features were updated
    if 'aatype' not in updated_feature_dict or not jnp.any(updated_feature_dict['aatype'] != feature_dict['aatype']):
        logging.warning("'aatype' feature might not have been updated correctly.")
    if 'msa' not in updated_feature_dict:
         # Check if msa_profile was updated instead
         if 'msa_profile' not in updated_feature_dict or ('msa_profile' in feature_dict and not jnp.any(updated_feature_dict['msa_profile'] != feature_dict['msa_profile'])):
              logging.warning("Neither 'msa' dict nor 'msa_profile' seemed to be updated correctly.")

    logging.debug(f"Finished updating features from logits.")
    return updated_feature_dict 
import numpy as np
import jax
import jax.numpy as jnp
from absl import logging
from alphafold3.common import folding_input

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
    # --- End Log ---

    # Get Chain ID to Residue Mapping - use token_features.asym_id instead of chain_id_per_residue
    chain_id_key = 'token_features.asym_id'
    
    # Check if token_features.asym_id is available
    if chain_id_key in feature_dict:
        int_asym_ids = feature_dict[chain_id_key]
    # Fallback checks for other possible keys
    elif 'asym_id' in feature_dict:
        int_asym_ids = feature_dict['asym_id']
    else:
        logging.error("Available feature dict keys: " + str(list(feature_dict.keys())))
        raise KeyError(f"Feature '{chain_id_key}' or 'asym_id' needed for binder setup not found.")
        
    # --- Create mapping from integer asym_id to string chain ID --- 
    # Robust approach: Map based on residue ranges from input chains
    int_to_str_chain_map = {}
    current_index = 0
    all_found_int_ids = set()
    for chain in fold_input.chains:
        if not hasattr(chain, 'id') or not hasattr(chain, 'sequence'):
            logging.warning(f"Skipping chain in mapping creation as it lacks id or sequence: {chain}")
            continue
        chain_id_str = chain.id
        chain_len = len(chain.sequence)
        end_index = current_index + chain_len
        
        # Ensure indices are within the bounds of int_asym_ids
        if end_index > len(int_asym_ids):
            logging.error(f"Chain {chain_id_str} end index ({end_index}) exceeds feature length ({len(int_asym_ids)}). Cannot map correctly.")
            break # Stop mapping if lengths don't match
            
        # Find the unique integer ID(s) in this residue range
        ids_in_range = np.unique(int_asym_ids[current_index:end_index])
        
        if len(ids_in_range) == 1:
            int_id = int(ids_in_range[0]) # Convert numpy int to Python int
            if int_id in int_to_str_chain_map and int_to_str_chain_map[int_id] != chain_id_str:
                logging.warning(f"Integer ID {int_id} already mapped to {int_to_str_chain_map[int_id]}, but also found for chain {chain_id_str}. Overwriting.")
            int_to_str_chain_map[int_id] = chain_id_str
            all_found_int_ids.add(int_id)
        elif len(ids_in_range) == 0:
             logging.warning(f"No integer asym_ids found in range [{current_index}:{end_index}] for chain {chain_id_str}.")
        else:
            logging.warning(f"Multiple integer asym_ids ({ids_in_range}) found in range [{current_index}:{end_index}] for chain {chain_id_str}. Using first ID {ids_in_range[0]}. Mapping might be incorrect.")
            int_id = int(ids_in_range[0])
            if int_id in int_to_str_chain_map and int_to_str_chain_map[int_id] != chain_id_str:
                 logging.warning(f"Integer ID {int_id} (first of multiple) already mapped to {int_to_str_chain_map[int_id]}, but also found for chain {chain_id_str}. Overwriting.")
            int_to_str_chain_map[int_id] = chain_id_str
            all_found_int_ids.add(int_id)
            
        current_index = end_index

    # Log any unmapped integer IDs found in the full array
    all_int_ids_in_features = set(np.unique(int_asym_ids).astype(int))
    unmapped_ids = all_int_ids_in_features - all_found_int_ids
    if unmapped_ids:
        logging.warning(f"The following integer asym_ids exist in features but were not mapped based on input chain ranges: {unmapped_ids}. These might belong to padding or indicate an issue.")

    # unique_int_ids = sorted(np.unique(int_asym_ids))
    # original_chain_ids = [chain.id for chain in fold_input.chains if hasattr(chain, 'id')]
    
    # if len(unique_int_ids) != len(original_chain_ids):
    #     logging.warning(
    #         f"Mismatch between unique integer asym_ids ({len(unique_int_ids)}) "
    #         f"and number of chains in input ({len(original_chain_ids)}). "
    #         f"Mapping might be incorrect."
    #     )
    #     # Attempt simple 1-to-1 mapping anyway, might work for simple cases
    #     int_to_str_chain_map = {int_id: str_id for int_id, str_id in zip(unique_int_ids, original_chain_ids)}
    # else:
    #     int_to_str_chain_map = {int_id: str_id for int_id, str_id in zip(unique_int_ids, original_chain_ids)}
    logging.info(f"Created integer-to-string chain map: {int_to_str_chain_map}")
    # --- End Mapping --- 
    
    target_indices, binder_indices = get_residue_indices(
        int_asym_ids, target_chains, binder_chains, int_to_str_chain_map
    )
    
    # Add Design Mask
    # Check for the actual key 'seq_mask' first, then fallbacks
    if 'seq_mask' in feature_dict:
        seq_mask_key = 'seq_mask'
    elif 'token_features.mask' in feature_dict:
        seq_mask_key = 'token_features.mask'
        logging.warning("Using fallback key 'token_features.mask' for sequence mask.")
    elif 'sequence_mask' in feature_dict: # Least likely, but check just in case
        seq_mask_key = 'sequence_mask'
        logging.warning("Using fallback key 'sequence_mask' for sequence mask.")
    else:
        # Print available keys to help debug
        logging.error("Available feature dict keys: " + str(list(feature_dict.keys())))
        raise KeyError("Could not find a valid sequence mask key ('seq_mask', 'token_features.mask', or 'sequence_mask').")
    
    logging.info(f"Using key '{seq_mask_key}' for design mask.")
    design_mask = np.zeros_like(feature_dict[seq_mask_key])
    design_mask[binder_indices] = 1
    feature_dict['design_mask'] = design_mask
    logging.info(f"Added 'design_mask' with {int(design_mask.sum())} designable positions.")
    
    # Store Initial Coords (for target FAPE loss)
    # Use template_atom_positions as the source for initial coordinates
    atom_pos_key = 'template_atom_positions' 
    if atom_pos_key in feature_dict:
        initial_coords = feature_dict[atom_pos_key].copy()
        feature_dict['initial_coords'] = initial_coords
        logging.info(f"Stored 'initial_coords' (from {atom_pos_key}) for potential target FAPE loss.")
    else:
        logging.warning(f"Could not find '{atom_pos_key}' to store initial coordinates.")
    
    # Mask Binder MSA
    # Use 'msa' key instead of 'msa_feat'
    msa_key = 'msa' 
    msa_mask_key = 'msa_mask'
    if msa_key in feature_dict and msa_mask_key in feature_dict:
        logging.info(f"Masking MSA features (key: '{msa_key}') for binder residues.")
        try:
            # --- Log MSA Shape ---
            logging.info(f"Shape of feature_dict['{msa_key}']: {feature_dict[msa_key].shape}")
            logging.info(f"Shape of feature_dict['{msa_mask_key}']: {feature_dict[msa_mask_key].shape}")
            # --- End Log ---
            # Ensure binder_indices are valid for the sequence length dimension
            seq_len = feature_dict[msa_key].shape[1]
            valid_binder_indices = binder_indices[binder_indices < seq_len]
            
            if len(valid_binder_indices) > 0:
                # Corrected 2D indexing
                feature_dict[msa_key][:, valid_binder_indices] = 0 
                feature_dict[msa_mask_key][:, valid_binder_indices] = 0
            else:
                logging.warning("No valid binder indices found within MSA sequence length.")
        except IndexError:
            logging.error("Error masking binder MSA - check dimensions and indices.")
            raise
    else:
        logging.warning(f"Could not find '{msa_key}' or '{msa_mask_key}' to mask binder MSA.")
    
    logging.info("Binder feature setup complete.")
    return feature_dict, target_indices, binder_indices

def update_features_from_logits(feature_dict, binder_indices, binder_seq_logits):
    """Updates feature_dict in place based on current binder sequence logits.
    
    Args:
        feature_dict: Dictionary of model features.
        binder_indices: Indices of the binder residues.
        binder_seq_logits: Sequence logits for the binder regions.
        
    Returns:
        Updated feature dictionary.
    """
    # --- Log Feature Dict Keys During Update ---
    #logging.info(f"Keys in feature_dict during update: {list(feature_dict.keys())}")
    # --- End Log ---
    if 'design_mask' not in feature_dict:
        raise ValueError("'design_mask' must be in feature_dict before calling update.")
    
    # Calculate probabilities and aatype
    binder_probs = jax.nn.softmax(binder_seq_logits, axis=-1)
    binder_aatype = jnp.argmax(binder_probs, axis=-1)
    
    # Update 'aatype' feature
    aatype_key = 'aatype'
    if aatype_key in feature_dict:
        target_dtype_aatype = feature_dict[aatype_key].dtype
        logging.debug(f"Updating '{aatype_key}' (dtype: {target_dtype_aatype}) with binder_aatype (dtype: {binder_aatype.dtype})")
        feature_dict[aatype_key] = feature_dict[aatype_key].at[binder_indices].set(binder_aatype.astype(target_dtype_aatype))
    else:
        logging.warning(f"Key '{aatype_key}' not found for updating.")
    
    # Update 'msa' first sequence (target sequence representation)
    msa_key = 'msa'
    if msa_key in feature_dict:
        target_dtype_msa = feature_dict[msa_key].dtype
        logging.debug(f"Updating '{msa_key}' (dtype: {target_dtype_msa}) with binder_aatype (dtype: {binder_aatype.dtype})")
        # Update the first row (target sequence) for binder positions with amino acid indices (aatype)
        # Ensure binder_aatype has the correct shape (should be 1D)
        if binder_aatype.ndim == 1:
             feature_dict[msa_key] = feature_dict[msa_key].at[0, binder_indices].set(binder_aatype.astype(target_dtype_msa))
        else:
             logging.error(f"binder_aatype has unexpected shape {binder_aatype.shape} during MSA update. Expected 1D.")
    else:
        logging.warning(f"Key '{msa_key}' not found for updating.")
    
    return feature_dict 
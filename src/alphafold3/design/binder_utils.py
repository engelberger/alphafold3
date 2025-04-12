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

def update_features_from_logits(feature_dict, binder_indices, binder_seq_logits):
    """Updates feature_dict in place based on current binder sequence logits.
    
    Args:
        feature_dict: Dictionary of model features.
        binder_indices: Indices of the binder residues.
        binder_seq_logits: Sequence logits for the binder regions.
        
    Returns:
        Updated feature dictionary.
    """
    from absl import logging
    # Ensure everything is JAX arrays
    binder_indices = jnp.asarray(binder_indices)
    binder_seq_logits = jnp.asarray(binder_seq_logits)
    
    if 'design_mask' not in feature_dict:
        logging.warning("'design_mask' not found in feature_dict. Adding default mask based on binder_indices.")
        design_mask = jnp.zeros((feature_dict.get('aatype', jnp.zeros((100,))).shape[0],), dtype=jnp.int32)
        # Use dynamic_update_slice or functional scatter update instead of at[] for better JAX compatibility
        feature_dict['design_mask'] = design_mask
    
    # Calculate probabilities and aatype
    binder_probs = jax.nn.softmax(binder_seq_logits, axis=-1)
    binder_aatype = jnp.argmax(binder_probs, axis=-1)
    
    # Make a copy of the feature dict to avoid in-place mutation issues with JAX
    updated_feature_dict = {}
    
    # Process each key in the feature dict
    for k, v in feature_dict.items():
        if k == 'aatype':
            # Update 'aatype' feature using scatter
            target_dtype = v.dtype
            updated_v = v.copy()  # Make a copy to avoid in-place operations
            
            def update_indices(val, idx):
                # Use functional scatter
                updated_val = val.at[idx].set(binder_aatype.astype(target_dtype))
                return updated_val
            
            # Update the aatype features at binder indices
            updated_feature_dict[k] = update_indices(v, binder_indices)
            
            logging.debug(f"Updated {k} with shape {updated_feature_dict[k].shape}")
        
        elif k == 'msa' and v.shape[0] > 0:
            # Update msa for the first row (target sequence) using scatter
            target_dtype = v.dtype
            updated_v = v.copy()  # Make a copy to avoid in-place operations
            
            # Define update function for MSA - uses functional scatter approach
            def update_msa_indices(val, row_idx, col_indices):
                # Update first row of MSA at binder positions
                updated_val = val.at[row_idx, col_indices].set(binder_aatype.astype(target_dtype))
                return updated_val
            
            # Update the first row of MSA at binder positions
            updated_feature_dict[k] = update_msa_indices(v, 0, binder_indices)
            
            logging.debug(f"Updated {k} with shape {updated_feature_dict[k].shape}")
        else:
            # Copy other features as-is
            updated_feature_dict[k] = v
    
    return updated_feature_dict 
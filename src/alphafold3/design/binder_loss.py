import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

def get_binder_plddt_loss(result, binder_indices):
    """Calculate loss based on pLDDT values for binder residues.
    
    Args:
        result: AlphaFold model result dictionary or confidence head output.
        binder_indices: Indices of the binder residues.
        
    Returns:
        Negative average pLDDT (to minimize as a loss function).
    """
    # Handle different output formats
    if isinstance(result, dict) and 'predicted_lddt' in result:
        plddt = result['predicted_lddt']
    else:
        logging.warning("Could not find predicted_lddt in result. Format may be incorrect.")
        return 0.0
    
    # plddt shape is likely (num_samples, num_residues_padded, num_atoms)
    if len(plddt.shape) != 3:
        logging.error(f"Unexpected pLDDT shape: {plddt.shape}. Expected 3 dimensions (samples, residues, atoms?).")
        # Attempt to proceed assuming shape is (residues, atoms) or just (residues)
        if len(plddt.shape) == 2:
            binder_plddt = plddt[binder_indices, :]
            binder_plddt_mean = jnp.mean(binder_plddt, axis=-1)
        elif len(plddt.shape) == 1:
            binder_plddt_mean = plddt[binder_indices]
        else:
            return 0.0 # Cannot handle other shapes
    else:
        # Correct indexing: axis 1 for residues
        # binder_indices should be valid for the padded dimension if features were padded
        binder_plddt_samples_atoms = plddt[:, binder_indices, :]
        
        # Average over atoms (axis 2) and samples (axis 0)
        binder_plddt_mean = jnp.mean(binder_plddt_samples_atoms, axis=(0, 2))
    
    # Return negative average pLDDT (as we want to maximize pLDDT)
    # Ensure the result is scalar
    return -jnp.mean(binder_plddt_mean)

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
    if isinstance(result, dict):
        # Check various possible PAE keys
        for pae_key in ['full_pae', 'pae']:
            if pae_key in result:
                pae = result[pae_key]
                break
        else:
            logging.warning("Could not find PAE data in result. Format may be incorrect.")
            return 0.0
    else:
        logging.warning("Result is not a dictionary. Cannot extract PAE.")
        return 0.0
    
    # Some models may have batched PAE matrices
    if len(pae.shape) > 2:
        # Average across batches if there are multiple samples
        pae = jnp.mean(pae, axis=0)
    
    # Extract interface PAE (target to binder interactions)
    interface_pae = pae[target_indices][:, binder_indices]
    
    # Return average PAE at the interface (lower is better)
    return jnp.mean(interface_pae)

def get_contact_loss(result, target_indices, binder_indices):
    """Calculate loss based on contact probability.
    
    Args:
        result: AlphaFold model result or distogram result.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.
        
    Returns:
        Negative average contact probability at the interface.
    """
    # Check if this is from full model or just distogram output
    if 'contact_probs' in result:
        contact_probs = result['contact_probs']
    else:
        # For full model result
        contact_probs = result.get('distogram', {}).get('contact_probs', None)
        
    if contact_probs is None:
        logging.warning("Contact probabilities not found in result dictionary")
        return 0.0
    
    # Extract interface contacts (target to binder interactions)
    interface_contacts = contact_probs[target_indices][:, binder_indices]
    
    # Return negative average contact probability (as we want to maximize contacts)
    return -jnp.mean(interface_contacts)

def get_target_fape_loss(result, initial_coords, target_indices):
    """Calculate loss to maintain target structure.
    
    Args:
        result: AlphaFold model result dictionary.
        initial_coords: Initial coordinates of the structure.
        target_indices: Indices of the target residues.
        
    Returns:
        FAPE loss for target residues.
    """
    # Get final coordinates
    final_coords = result['final_atom_positions']
    
    # Calculate RMSD between initial and final coordinates for target
    target_initial_coords = initial_coords[target_indices]
    target_final_coords = final_coords[target_indices]
    
    # Create mask for valid atoms
    valid_mask = jnp.any(target_initial_coords != 0, axis=-1)
    
    # Calculate squared distances
    squared_diff = jnp.sum(
        jnp.square(target_final_coords - target_initial_coords) * valid_mask[..., None],
        axis=-1
    )
    
    # Return average RMSD for valid atoms
    return jnp.sum(squared_diff) / (jnp.sum(valid_mask) + 1e-8)

def get_binder_seq_entropy_loss(binder_seq_logits):
    """Calculate sequence entropy loss to encourage diversity.
    
    Args:
        binder_seq_logits: Sequence logits for the binder.
        
    Returns:
        Negative sequence entropy.
    """
    probs = jax.nn.softmax(binder_seq_logits, axis=-1)
    log_probs = jnp.log(probs + 1e-8)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    
    # Return negative entropy (as we want to maximize entropy for diversity)
    return -jnp.mean(entropy)

def get_distogram_contact_loss(distogram_result, target_indices, binder_indices):
    """Calculate loss based on distogram contacts.
    
    Args:
        distogram_result: Distogram head output dictionary.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.
        
    Returns:
        Negative average contact probability at the interface.
    """
    # Extract the contact probabilities at the specified cutoff
    # In AlphaFold3, the distogram output already provides 'contact_probs'
    if 'contact_probs' in distogram_result:
        contact_probs = distogram_result['contact_probs']
    else:
        # If direct contact probabilities aren't available, compute them from the distogram
        logging.warning("Direct contact_probs not found in distogram output, using fallback method")
        bin_edges = distogram_result.get('bin_edges', None)
        if bin_edges is None:
            logging.error("Cannot compute contact probabilities: bin_edges not found in distogram output")
            return 0.0
            
        # Get the distogram probabilities
        distogram_probs = distogram_result.get('probs', distogram_result.get('logits', None))
        if distogram_probs is None:
            logging.error("Cannot compute contact probabilities: no probability or logit data found")
            return 0.0
            
        # Find which bins correspond to distances < 8Å
        contact_threshold = 8.0  # standard contact threshold
        contact_bins = (bin_edges < contact_threshold).sum()
        
        # Sum probabilities for bins that correspond to contact
        contact_probs = jnp.sum(distogram_probs[..., :contact_bins], axis=-1)
    
    # Extract interface contacts (target to binder interactions)
    interface_contacts = contact_probs[target_indices][:, binder_indices]
    
    # Return negative average contact probability (as we want to maximize contacts)
    return -jnp.mean(interface_contacts)

def calculate_gradient_binder_loss(result, feature_dict, target_indices, binder_indices, binder_seq_logits, design_params):
    """Calculate loss for gradient-based binder design.
    
    Args:
        result: AlphaFold model result dictionary.
        feature_dict: Feature dictionary.
        target_indices: Indices of the target residues.
        binder_indices: Indices of the binder residues.
        binder_seq_logits: Sequence logits for the binder.
        design_params: Dictionary of design parameters.
        
    Returns:
        Tuple of (total_loss, loss_breakdown).
    """
    losses = {}
    weights = design_params.get("weights", {})
    
    # Calculate losses
    if weights.get('plddt', 0.0) > 0:
        losses['plddt'] = get_binder_plddt_loss(result, binder_indices)
    
    if weights.get('pae_inter', 0.0) > 0:
        losses['pae_inter'] = get_interface_pae_loss(result, target_indices, binder_indices)
    
    if weights.get('contact', 0.0) > 0:
        losses['contact'] = get_contact_loss(result, target_indices, binder_indices)
    
    if weights.get('seq_entropy', 0.0) > 0:
        losses['seq_entropy'] = get_binder_seq_entropy_loss(binder_seq_logits)
    
    if weights.get('fape_target', 0.0) > 0 and 'initial_coords' in feature_dict:
        losses['fape_target'] = get_target_fape_loss(
            result, feature_dict['initial_coords'], target_indices
        )

    # Calculate weighted total loss
    total_loss = 0.0
    for k, v in losses.items():
        weighted_loss = weights.get(k, 0.0) * v
        total_loss += weighted_loss
        losses[k] = v  # Store unweighted loss
    
    return total_loss, losses

def calculate_boltz_binder_loss(partial_result, feature_dict, target_indices, binder_indices, binder_seq_logits, design_params):
    """Calculate loss for BoltzDesign1-like binder design.
    
    Args:
        partial_result: Partial model result (distogram and confidence outputs directly in dict).
        feature_dict: Feature dictionary.
        target_indices: Indices of the target residues. 
        binder_indices: Indices of the binder residues.
        binder_seq_logits: Sequence logits for the binder.
        design_params: Dictionary of design parameters.
        
    Returns:
        Tuple of (total_loss, loss_breakdown).
    """
    from absl import logging
    
    # --- Validate inputs ---
    logging.info("==== STARTING BOLTZ BINDER LOSS CALCULATION ====")
    logging.info(f"Target indices: {target_indices.shape}, Binder indices: {binder_indices.shape}")
    logging.info(f"Binder logits shape: {binder_seq_logits.shape}")
    
    # Check for empty indices
    if len(target_indices) == 0:
        logging.error("Target indices array is empty! Cannot calculate loss.")
        raise ValueError("Empty target_indices array")
        
    if len(binder_indices) == 0:
        logging.error("Binder indices array is empty! Cannot calculate loss.")
        raise ValueError("Empty binder_indices array")

    # Log partial_result keys
    logging.info(f"partial_result keys: {list(partial_result.keys())}")
    
    # Log design weights
    weights = design_params.get("weights", {})
    logging.info(f"Design weights: {weights}")
    
    # Initialize losses dictionary with zero values for all potential loss terms
    # This ensures consistent dictionary structure regardless of which losses are calculated
    losses = {
        'distogram': jnp.array(0.0, dtype=jnp.float32),
        'plddt': jnp.array(0.0, dtype=jnp.float32),
        'pae': jnp.array(0.0, dtype=jnp.float32),
        'seq_entropy': jnp.array(0.0, dtype=jnp.float32),
        # Initialize seq_one_hot to ensure consistent structure with loss_fn_for_grad_boltz
        'seq_one_hot': jnp.array(0.0, dtype=jnp.float32)
    }
    
    # Calculate losses from partial outputs (no backprop through structure module)
    if weights.get('distogram', 0.0) > 0:
        if 'distogram' in partial_result:
            distogram_result = partial_result['distogram']
            try:
                logging.info(f"Calculating distogram loss with shapes - target: {target_indices.shape}, binder: {binder_indices.shape}")
                distogram_loss = get_distogram_contact_loss(
                    distogram_result, target_indices, binder_indices
                )
                losses['distogram'] = jnp.where(
                    jnp.isnan(distogram_loss) | jnp.isinf(distogram_loss),
                    jnp.array(0.0, dtype=distogram_loss.dtype),
                    distogram_loss
                )
            except Exception as e:
                logging.error(f"Error in distogram loss calculation (not logging full error due to JIT): {type(e)}")
                losses['distogram'] = jnp.array(0.0, dtype=jnp.float32)
        else:
            logging.warning("Distogram weight > 0 but 'distogram' not in partial_result, skipping")
            pass
    
    # Check for confidence weight > 0 before accessing confidence keys
    if weights.get('confidence', 0.0) > 0:
        confidence_calculated = False # Flag to track if any confidence loss was calculated
        # Access confidence keys directly from partial_result
        if 'predicted_lddt' in partial_result:
            try:
                logging.info(f"Calculating pLDDT loss with binder shape: {binder_indices.shape}")
                # Pass partial_result directly as it contains predicted_lddt
                plddt_loss = get_binder_plddt_loss(partial_result, binder_indices)
                losses['plddt'] = jnp.where(
                    jnp.isnan(plddt_loss) | jnp.isinf(plddt_loss),
                    jnp.array(0.0, dtype=plddt_loss.dtype),
                    plddt_loss
                )
                confidence_calculated = True
            except Exception as e:
                logging.error(f"Error in pLDDT loss calculation (not logging full error due to JIT): {type(e)}")
                pass
        else:
            logging.warning("Confidence weight > 0 but 'predicted_lddt' not in partial_result")

        if 'full_pae' in partial_result:
            try:
                logging.info(f"Calculating PAE loss with shapes - target: {target_indices.shape}, binder: {binder_indices.shape}")
                 # Pass partial_result directly as it contains full_pae
                pae_loss = get_interface_pae_loss(partial_result, target_indices, binder_indices)
                losses['pae'] = jnp.where(
                    jnp.isnan(pae_loss) | jnp.isinf(pae_loss),
                    jnp.array(0.0, dtype=pae_loss.dtype),
                    pae_loss
                )
                confidence_calculated = True
            except Exception as e:
                logging.error(f"Error in PAE loss calculation (not logging full error due to JIT): {type(e)}")
                pass
        else:
             logging.warning("Confidence weight > 0 but 'full_pae' not in partial_result")

        if not confidence_calculated:
             logging.warning("Confidence weight > 0 but required keys ('predicted_lddt', 'full_pae') were missing from partial_result. Confidence loss is 0.")

    if weights.get('seq_entropy', 0.0) > 0:
        try:
            logging.info(f"Calculating sequence entropy loss with logits shape: {binder_seq_logits.shape}")
            entropy_loss = get_binder_seq_entropy_loss(binder_seq_logits)
            losses['seq_entropy'] = jnp.where(
                jnp.isnan(entropy_loss) | jnp.isinf(entropy_loss),
                jnp.array(0.0, dtype=entropy_loss.dtype),
                entropy_loss
            )
        except Exception as e:
            logging.error(f"Error in sequence entropy loss calculation (not logging full error due to JIT): {type(e)}")
            pass

    # Calculate weighted total loss with safety checks
    total_loss = jnp.array(0.0, dtype=jnp.float32)
    for k, v in losses.items():
        # Skip seq_one_hot which will be handled separately in the loss_fn_for_grad_boltz
        if k == 'seq_one_hot':
            continue
        weight = weights.get(k, 0.0)
        weighted_loss = weight * v
        # In JAX, use functional style for accumulation
        total_loss = total_loss + weighted_loss
    
    # Don't log directly due to JIT
    # Instead, return the raw values to be logged outside JIT
    logging.info("Completed loss calculation")
    
    return total_loss, losses 
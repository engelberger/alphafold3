"""Binder design protocols for AlphaFold 3."""

import time
import functools
from typing import Dict, Any, Tuple, List, Callable, Optional, Sequence
import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import model
from alphafold3.model import features
from alphafold3.model.network import distogram_head
from alphafold3.model.network import confidence_head
from alphafold3.design import binder_utils
from alphafold3.design import binder_loss


class BinderDesigner:
    """Class to handle binder design using AlphaFold 3."""

    def __init__(
        self,
        model_runner: Any,
        ccd: chemical_components.Ccd,
        design_params: Dict[str, Any],
    ):
        """Initialize the binder designer.
        
        Args:
            model_runner: ModelRunner instance.
            ccd: Chemical component dictionary.
            design_params: Dictionary of design parameters.
        """
        self.model_runner = model_runner
        self.ccd = ccd
        self.design_params = design_params
        self.protocol = design_params.get("protocol", "binder_gradient")
        
        logging.info(f"Initialized BinderDesigner with protocol: {self.protocol}")
        logging.info(f"Design parameters: {design_params}")
    
    def design_binder(
        self, 
        fold_input: folding_input.Input,
        feature_dict: features.BatchDict,
        rng_key: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """Design a binder protein using the specified protocol.
        
        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The feature dictionary for the model.
            rng_key: JAX random key.
            
        Returns:
            Tuple of (design_results, final_feature_dict).
        """
        target_chains = self.design_params["target_chains"]
        binder_chains = self.design_params["binder_chains"]
        
        # Modify feature_dict for binder design
        feature_dict, target_indices, binder_indices = binder_utils.setup_binder_features(
            feature_dict, target_chains, binder_chains, fold_input=fold_input
        )
        
        # Convert necessary numpy arrays in feature_dict to JAX arrays for gradient updates
        if 'aatype' in feature_dict:
            feature_dict['aatype'] = jnp.asarray(feature_dict['aatype'])
        if 'msa' in feature_dict:
            feature_dict['msa'] = jnp.asarray(feature_dict['msa'])
        # Add other keys here if they cause similar errors later
        logging.info("Converted 'aatype' and 'msa' in feature_dict to JAX arrays.")
        
        # Choose the appropriate design protocol
        if self.protocol == "binder_gradient":
            return self._design_binder_gradient(
                fold_input, feature_dict, target_indices, binder_indices, rng_key
            )
        elif self.protocol == "binder_boltz":
            return self._design_binder_boltz(
                fold_input, feature_dict, target_indices, binder_indices, rng_key
            )
        else:
            raise ValueError(f"Unknown design protocol: {self.protocol}")
    
    def _design_binder_gradient(
        self,
        fold_input: folding_input.Input,
        feature_dict: features.BatchDict,
        target_indices: np.ndarray,
        binder_indices: np.ndarray,
        rng_key: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """Gradient-based binder design using full model.
        
        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The feature dictionary for the model.
            target_indices: Indices of target residues.
            binder_indices: Indices of binder residues.
            rng_key: JAX random key.
            
        Returns:
            Tuple of (design_results, final_feature_dict).
        """
        logging.info("Starting gradient-based binder design...")
        design_start_time = time.time()
        
        # Initialize sequence logits randomly
        num_residue_types = 20  # Standard amino acids
        binder_seq_logits = jnp.zeros((len(binder_indices), num_residue_types))
        binder_seq_logits = 0.01 * jax.random.normal(
            jax.random.PRNGKey(0), binder_seq_logits.shape
        )
        
        # Setup optimizer
        lr = self.design_params.get("lr", 0.1)
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(binder_seq_logits)
        
        # --- Filter out non-JAX compatible types from feature_dict_template --- 
        # Identify keys holding object arrays (like Structure objects)
        keys_to_remove = [k for k, v in feature_dict.items() if isinstance(v, np.ndarray) and v.dtype == object]
        logging.info(f"Filtering object-dtype keys from feature_dict for JIT: {keys_to_remove}")
        # Create a dictionary containing only JAX-compatible data
        feature_dict_jax = {k: v for k, v in feature_dict.items() if k not in keys_to_remove}
        # --------------------------------------------------------------------------
        
        # Setup logging and trajectory storage
        trajectory = {
            "loss": [],
            "losses": [],
            "step": [],
            "time": [],
            "sequences": [],
        }
        
        # Define loss function for gradient calculation
        def loss_fn_for_grad(curr_binder_logits, feature_dict_template, target_indices, 
                            binder_indices, design_params_static, model_runner_obj, rng_key):
            # Update features based on logits
            updated_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_template.copy(), binder_indices, curr_binder_logits
            )
            
            # Run full forward pass
            result = model_runner_obj.run_inference(updated_feature_dict, rng_key)
            
            # --- Debugging: Log shapes --- 
            #try:
            #    plddt_shape = result.get('predicted_lddt', 'Key not found').shape if 'predicted_lddt' in result else 'Key not found'
            #    binder_indices_shape = binder_indices.shape
            #    logging.info(f"[Debug] predicted_lddt shape: {plddt_shape}")
            #    logging.info(f"[Debug] binder_indices shape: {binder_indices_shape}")
            #    logging.info(f"[Debug] binder_indices[:5]: {binder_indices[:5]}")
            #    logging.info(f"[Debug] binder_indices[-5:]: {binder_indices[-5:]}")
            #except Exception as e:
            #    logging.error(f"[Debug] Error logging shapes: {e}")
            # --- End Debugging ---
            
            # Calculate loss
            total_loss, loss_breakdown = binder_loss.calculate_gradient_binder_loss(
                result, updated_feature_dict, target_indices, binder_indices, 
                curr_binder_logits, design_params_static
            )
            
            return total_loss, (loss_breakdown, result, updated_feature_dict)
        
        grad_fn = jax.value_and_grad(loss_fn_for_grad, has_aux=True)
        
        # Standard optimization loop
        best_loss = float('inf')
        best_logits = None
        best_feature_dict = None
        steps = self.design_params.get("steps", 200)
        
        for step in range(steps):
            step_start_time = time.time()
            step_key, rng_key = jax.random.split(rng_key)
            
            # Compute gradients
            (loss, (losses, result, updated_feature_dict)), grads = grad_fn(
                binder_seq_logits, feature_dict_jax, target_indices, binder_indices, 
                self.design_params, self.model_runner, step_key
            )
            
            # Update logits
            updates, opt_state = optimizer.update(grads, opt_state, binder_seq_logits)
            binder_seq_logits = optax.apply_updates(binder_seq_logits, updates)
            
            # Convert to concrete values for logging
            loss_val = float(loss)
            losses_val = {k: float(v) for k, v in losses.items()}
            
            # Log progress
            step_time = time.time() - step_start_time
            if step % 10 == 0 or step == steps - 1:
                probs = jax.nn.softmax(binder_seq_logits, axis=-1)
                aa_indices = jnp.argmax(probs, axis=-1)
                
                # Convert to amino acid sequence using the data_constants mapping
                import string
                aa_letters = list(string.ascii_uppercase)[:20]  # Simple A-T for 20 amino acids
                current_seq = ''.join([aa_letters[idx] for idx in aa_indices])
                
                logging.info(f"Step {step}/{steps}: loss={loss_val:.4f}, time={step_time:.2f}s")
                logging.info(f"Losses: {losses_val}")
                
                trajectory["sequences"].append(current_seq)
            
            # Store in trajectory
            trajectory["loss"].append(loss_val)
            trajectory["losses"].append(losses_val)
            trajectory["step"].append(step)
            trajectory["time"].append(step_time)
            
            # Track best loss
            if loss_val < best_loss:
                best_loss = loss_val
                best_logits = binder_seq_logits
                best_feature_dict = updated_feature_dict
        
        # Final sequence
        final_probs = jax.nn.softmax(best_logits, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)
        
        # Prepare results
        design_results = {
            "protocol": "binder_gradient",
            "target_indices": target_indices,
            "binder_indices": binder_indices,
            "best_loss": best_loss,
            "trajectory": trajectory,
            "design_time": time.time() - design_start_time,
            "final_seq_logits": best_logits,
            "final_aa_indices": final_aa_indices,
        }
        
        logging.info(f"Gradient-based binder design completed in {design_results['design_time']:.2f}s")
        logging.info(f"Best loss: {best_loss:.4f}")
        
        return design_results, best_feature_dict
    
    def _boltz_forward_pass(self, feature_dict, rng_key, stop_gradient=True):
        """Modified forward pass for BoltzDesign1 approach.
        
        This runs only the Pairformer and Confidence modules without backpropagating
        through the Structure (Diffusion) module.
        
        Args:
            feature_dict: The feature dictionary for the model.
            rng_key: JAX random key.
            stop_gradient: Whether to stop gradients at the structure module.
            
        Returns:
            Partial results from the forward pass.
        """
        # Run the standard model inference but with a special mode parameter
        # that indicates this is a boltz design run (will stop_gradient at structure module)
        boltz_result = self.model_runner.run_inference(
            feature_dict, 
            rng_key,
            mode="boltz_design"  # Special mode to signal gradient stopping for BoltzDesign1
        )
        
        logging.info("Ran BoltzDesign forward pass successfully using main model with 'boltz_design' mode")
        return boltz_result
    
    def _design_binder_boltz(
        self,
        fold_input: folding_input.Input,
        feature_dict: features.BatchDict,
        target_indices: np.ndarray,
        binder_indices: np.ndarray,
        rng_key: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """BoltzDesign1-like binder design using partial model outputs.
        
        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The feature dictionary for the model.
            target_indices: Indices of target residues.
            binder_indices: Indices of binder residues.
            rng_key: JAX random key.
            
        Returns:
            Tuple of (design_results, final_feature_dict).
        """
        logging.info("Starting BoltzDesign1-like binder design...")
        design_start_time = time.time()
        
        # Keep original feature_dict as template
        feature_dict_template = feature_dict
        
        # Convert NumPy arrays to tuples for static JIT arguments
        target_indices_static_tuple = tuple(map(int, target_indices))
        binder_indices_static_tuple = tuple(map(int, binder_indices))
        logging.info("Converted target/binder indices to tuples for static JIT arguments.")
        
        # Initialize sequence logits randomly
        num_residue_types = 20  # Standard amino acids
        binder_seq_logits = jnp.zeros((len(binder_indices), num_residue_types))
        binder_seq_logits = 0.01 * jax.random.normal(
            jax.random.PRNGKey(0), binder_seq_logits.shape
        )
        
        # Get stage configuration
        stages = self.design_params.get("stages", [50, 50, 50, 50])  # Default 4 stages
        total_steps = sum(stages)
        
        # Setup optimizer
        lr = self.design_params.get("lr", 0.1)
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(binder_seq_logits)
        
        # --- Filter out non-JAX compatible types from feature_dict_template --- 
        # Identify keys holding object arrays (like Structure objects)
        keys_to_remove = [k for k, v in feature_dict_template.items() if isinstance(v, np.ndarray) and v.dtype == object]
        logging.info(f"Filtering object-dtype keys from feature_dict_template for JIT: {keys_to_remove}")
        # Create a dictionary containing only JAX-compatible data
        feature_dict_template_jax = {k: v for k, v in feature_dict_template.items() if k not in keys_to_remove}
        # --------------------------------------------------------------------------
        
        # --- Extract numerical weights from design_params --- 
        # Pass only this potentially hashable dictionary to the JITted function
        design_weights_static = self.design_params.get('weights', {})
        logging.info(f"Extracted static weights for JIT: {design_weights_static}")
        # Dictionaries are generally not hashable for JIT, treat weights as non-static.
        # stage_idx (1) must be static for control flow. Indices (5, 6) and runner (8) are also static.
        static_argnums_for_jit = (1, 5, 6, 8) # stage_idx, Indices (tuples), Runner
        # -----------------------------------------------------
        
        # Setup logging and trajectory storage
        trajectory = {
            "loss": [],
            "losses": [],
            "step": [],
            "time": [],
            "stage": [],
            "sequences": [],
        }
        
        # Define loss function for gradient calculation with the boltz approach
        def loss_fn_for_grad_boltz(
            curr_binder_logits,     # Arg 0 (Non-static, Differentiate)
            stage_idx,              # Arg 1 (Non-static, Varies)
            temp,                   # Arg 2 (Non-static, Varies)
            rng_key,                # Arg 3 (Non-static, Varies)
            # --- Static arguments ---
            feature_dict_template,  # Arg 4 (Static)
            target_indices_tuple,   # Arg 5 (Static - NOW A TUPLE)
            binder_indices_tuple,   # Arg 6 (Static - NOW A TUPLE)
            design_params_static,   # Arg 7 (Static)
            model_runner_obj        # Arg 8 (Static)
        ):
            # Convert tuples back to JAX arrays for use
            target_indices = jnp.array(target_indices_tuple, dtype=jnp.int32)
            binder_indices = jnp.array(binder_indices_tuple, dtype=jnp.int32)
            
            # Stage-specific sequence representation
            num_residue_types = curr_binder_logits.shape[-1]
            
            # Process sequence logits based on stage
            if stage_idx == 0:  # Stage 1: Exploration (T=1.0)
                temp = 1.0
                seq_probs = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
            elif stage_idx == 1:  # Stage 2: Transition
                # Use the temp parameter directly
                # Interpolation between logits and softmax probabilities
                seq_probs_logits = jax.nn.softmax(curr_binder_logits, axis=-1)
                seq_probs_softmax = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
                seq_probs = 0.5 * seq_probs_logits + 0.5 * seq_probs_softmax
            elif stage_idx == 2:  # Stage 3: Convergence (gradually decreasing temperature)
                # Initial temp for this stage
                temp = 0.5
                logging.info(f"Starting stage 3: Convergence with initial temperature {temp}")
                seq_probs = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
            elif stage_idx == 3:  # Stage 4: One-hot with straight-through estimator
                # Use the temp parameter directly
                # Softmax with very low temperature (approximates one-hot)
                seq_probs_soft = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
                # Get hard one-hot
                seq_probs_hard = jax.nn.one_hot(jnp.argmax(seq_probs_soft, axis=-1), num_residue_types)
                # Straight-through estimator (pass gradient through soft probabilities)
                seq_probs = jax.lax.stop_gradient(seq_probs_hard - seq_probs_soft) + seq_probs_soft
            else:
                raise ValueError(f"Invalid stage index: {stage_idx}")
            
            # Update features based on sequence probabilities - use JAX array binder_indices
            updated_feature_dict = binder_utils.update_features_from_logits(
                jax.tree_map(lambda x: x.copy() if isinstance(x, (jnp.ndarray, np.ndarray)) else x, feature_dict_template),
                binder_indices,  # Use JAX array here
                curr_binder_logits
            )
            
            # Run Boltz forward pass with gradient stopping at structure module
            step_key, _ = jax.random.split(rng_key)
            boltz_result = self._boltz_forward_pass(
                updated_feature_dict, step_key, stop_gradient=True
            )
            
            # Calculate BoltzDesign1 loss - use JAX arrays target_indices, binder_indices
            total_loss, loss_breakdown = binder_loss.calculate_boltz_binder_loss(
                boltz_result, updated_feature_dict, target_indices, binder_indices, 
                curr_binder_logits, design_params_static
            )
            
            # Apply additional sequence shaping in one-hot stage
            if stage_idx == 3:  # Final one-hot stage
                # Add straight-through cross-entropy to encourage convergence to one-hot
                seq_cross_entropy = -jnp.sum(seq_probs_hard * jax.nn.log_softmax(curr_binder_logits, axis=-1))
                seq_entropy_weight = design_params_static.get("weights", {}).get("seq_one_hot", 0.1)
                total_loss += seq_entropy_weight * seq_cross_entropy
                loss_breakdown["seq_one_hot"] = seq_cross_entropy
            
            return total_loss, (loss_breakdown, boltz_result, updated_feature_dict)
        
        # Define the gradient function directly on the loss function
        grad_loss_fn = jax.value_and_grad(loss_fn_for_grad_boltz, argnums=0, has_aux=True)
        
        # JIT the gradient function, marking static arguments
        # Arguments: 0=logits, 1=stage, 2=temp, 3=key, 4=features, 5=tgt idx tuple, 6=bnd idx tuple, 7=weights_dict, 8=runner
        # Use the correctly defined static_argnums_for_jit
        grad_fn = jax.jit(grad_loss_fn, static_argnums=static_argnums_for_jit)
        
        # Multi-stage optimization
        best_loss = float('inf')
        best_logits = None
        best_feature_dict = None
        current_step = 0
        
        for stage_idx, stage_steps in enumerate(stages):
            # Configure temperature schedule for each stage
            if stage_idx == 0:
                temp = 1.0  # Stage 1: Exploration (high temperature)
                logging.info(f"Starting stage 1: Exploration with temperature {temp}")
            elif stage_idx == 1:
                temp = 0.5  # Stage 2: Transition (intermediate temperature)
                logging.info(f"Starting stage 2: Transition with temperature {temp}")
            elif stage_idx == 2:
                # Stage 3: Convergence (gradually decreasing temperature)
                # Initial temp for this stage
                temp = 0.5
                logging.info(f"Starting stage 3: Convergence with initial temperature {temp}")
            else:
                # Stage 4: One-hot (very low temperature)
                temp = 0.01
                logging.info(f"Starting stage 4: One-hot refinement with temperature {temp}")
            
            for step in range(stage_steps):
                step_start_time = time.time()
                step_key, rng_key = jax.random.split(rng_key)
                
                # Update temperature for stage 3 (gradually decreasing)
                if stage_idx == 2:
                    # Temperature decreases quadratically from 0.5 to 0.01
                    t_stage = (step + 1) / stage_steps
                    temp = 0.01 + (0.5 - 0.01) * (1.0 - t_stage)**2
                
                # Call the JITted grad_fn with all arguments explicitly
                # Passing the tuples for static indices arguments and the filtered feature dict
                (loss, (losses, partial_result, updated_feature_dict)), grads = grad_fn(
                    binder_seq_logits,          # Arg 0
                    stage_idx,                  # Arg 1
                    temp,                       # Arg 2
                    step_key,                   # Arg 3
                    # --- Pass the filtered dict as non-static --- 
                    feature_dict_template_jax,  # Arg 4 
                    # --- Static arguments --- 
                    target_indices_static_tuple, # Arg 5 (Pass tuple)
                    binder_indices_static_tuple, # Arg 6 (Pass tuple)
                    design_weights_static,      # Arg 7 (Pass weights dict)
                    self.model_runner           # Arg 8 (Static)
                )
                
                # Update logits
                updates, opt_state = optimizer.update(grads, opt_state, binder_seq_logits)
                binder_seq_logits = optax.apply_updates(binder_seq_logits, updates)
                
                # Convert to concrete values for logging
                loss_val = float(loss)
                losses_val = {k: float(v) for k, v in losses.items()}
                
                # Log progress
                step_time = time.time() - step_start_time
                if step % 10 == 0 or step == stage_steps - 1:
                    probs = jax.nn.softmax(binder_seq_logits / temp, axis=-1)
                    aa_indices = jnp.argmax(probs, axis=-1)
                    
                    # Convert to amino acid sequence using the data_constants mapping
                    import string
                    aa_letters = list(string.ascii_uppercase)[:20]  # Simple A-T for 20 amino acids
                    current_seq = ''.join([aa_letters[idx] for idx in aa_indices])
                    
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: "
                                f"loss={loss_val:.4f}, time={step_time:.2f}s")
                    logging.info(f"Losses: {losses_val}")
                    
                    trajectory["sequences"].append(current_seq)
                
                # Store in trajectory
                trajectory["loss"].append(loss_val)
                trajectory["losses"].append(losses_val)
                trajectory["step"].append(current_step)
                trajectory["time"].append(step_time)
                trajectory["stage"].append(stage_idx)
                
                # Track best loss
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_logits = binder_seq_logits
                    best_feature_dict = updated_feature_dict
                
                current_step += 1
        
        # Final sequence - in the one-hot stage
        temp = 0.01  # Very low temperature for final output
        final_probs = jax.nn.softmax(best_logits / temp, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)
        
        # Prepare results
        design_results = {
            "protocol": "binder_boltz",
            "target_indices": target_indices,
            "binder_indices": binder_indices,
            "best_loss": best_loss,
            "trajectory": trajectory,
            "design_time": time.time() - design_start_time,
            "final_seq_logits": best_logits,
            "final_aa_indices": final_aa_indices,
        }
        
        logging.info(f"BoltzDesign1-like binder design completed in {design_results['design_time']:.2f}s")
        logging.info(f"Best loss: {best_loss:.4f}")
        
        return design_results, best_feature_dict
    
    def run_final_prediction(
        self,
        fold_input: folding_input.Input,
        design_results: Dict[str, Any],
        rng_key: jnp.ndarray,
    ) -> Tuple[model.ModelResult, features.BatchDict, folding_input.Input]:
        """Run a final prediction with the designed sequence (Simplified).
        
        Args:
            fold_input: The original input to AlphaFold.
            design_results: Results from the design process.
            rng_key: JAX random key.
            
        Returns:
            Tuple of (model_result, final_feature_dict, new_fold_input).
        """
        logging.info("Running final prediction with designed sequence (Simplified Method)...")
        
        # Get the final amino acid indices from the design results
        final_aa_indices = design_results["final_aa_indices"]
        
        # Convert to amino acid sequence using a proper mapping from data_constants
        try:
            from alphafold3.model import data_constants
            aa_types = data_constants.protein_restypes
        except (ImportError, AttributeError):
            # Fallback to standard alphabet if data_constants is not available
            aa_types = list("ACDEFGHIKLMNPQRSTVWY")
        
        # Map indices to amino acid sequences
        designed_sequence = ''.join([aa_types[idx] for idx in final_aa_indices])
        logging.info(f"Designed sequence ({len(designed_sequence)} aa): {designed_sequence[:50]}...")
        
        # Create a new fold_input with the designed sequence
        new_chains = []
        binder_chains = self.design_params["binder_chains"]
        
        # Create a mapping of chain ID to the new designed sequence
        binder_seq_by_chain = {}
        binder_start_idx = 0
        
        # First, build a mapping of chain ID to sequence length
        chain_lengths = {}
        for chain in fold_input.chains:
            chain_lengths[chain.id] = len(chain)
        
        # Build a mapping of binder chain ID to its designed sequence
        for binder_chain_id in binder_chains:
            if binder_chain_id in chain_lengths:
                chain_length = chain_lengths[binder_chain_id]
                # Extract the portion of the designed sequence for this chain
                binder_seq_by_chain[binder_chain_id] = designed_sequence[binder_start_idx:binder_start_idx + chain_length]
                binder_start_idx += chain_length
        
        # Create new chains with updated sequences where needed
        for chain in fold_input.chains:
            if isinstance(chain, folding_input.ProteinChain) and chain.id in binder_chains:
                # This is a binder chain - replace the sequence
                if chain.id in binder_seq_by_chain:
                    new_sequence = binder_seq_by_chain[chain.id]
                    # Create a new chain with the designed sequence
                    new_chain = folding_input.ProteinChain(
                        id=chain.id,
                        sequence=new_sequence,
                        ptms=chain.ptms,  # Keep the original PTMs
                        # Clear MSA and set templates to empty list to force re-search
                        unpaired_msa=None, 
                        paired_msa=None,
                        templates=[] # Use empty list instead of None
                    )
                    new_chains.append(new_chain)
                    logging.info(f"Replaced binder chain {chain.id} sequence and reset MSAs/templates.")
                else:
                    # Something went wrong - use the original chain
                    logging.warning(f"No designed sequence for binder chain {chain.id}, using original")
                    new_chains.append(chain)
            else:
                # This is a target chain or non-protein chain - keep it unchanged
                new_chains.append(chain)
        
        # Create a new Input object with the updated chains
        new_fold_input = folding_input.Input(
            name=fold_input.name + "_designed",
            chains=new_chains,
            rng_seeds=[fold_input.rng_seeds[0]],  # Use only the first seed
            bonded_atom_pairs=fold_input.bonded_atom_pairs,
            user_ccd=fold_input.user_ccd
        )
        
        logging.info("Created new input with designed binder sequence (Simplified Method)")
        
        # Use the best feature dict from design results and update it
        final_feature_dict = binder_utils.update_features_from_logits(
            design_results["best_feature_dict"].copy(),
            design_results["binder_indices"],
            design_results["final_seq_logits"]
        )
        
        # Run standard prediction
        logging.info("Running inference on designed sequence (Simplified Method)...")
        final_result = self.model_runner.run_inference(final_feature_dict, rng_key)
        
        logging.info("Final prediction complete (Simplified Method)")
        
        # Add designed sequence to the result metadata
        if isinstance(final_result, dict) and "metadata" in final_result:
            final_result["metadata"]["designed_sequence"] = designed_sequence
            final_result["metadata"]["designed_binder_chains"] = binder_chains
        
        return final_result, final_feature_dict, new_fold_input

    def run_complete_prediction(
        self,
        fold_input: folding_input.Input,
        design_results: Dict[str, Any],
        rng_key: jnp.ndarray,
        buckets: Sequence[int] | None = None,
        ref_max_modified_date: datetime.date | None = None,
        conformer_max_iterations: int | None = None,
    ) -> Tuple[model.ModelResult, features.BatchDict, folding_input.Input]:
        """Run a complete prediction pipeline with the designed sequence.
        
        This is a more comprehensive prediction method that creates a new input
        with the designed sequence and runs it through the full AlphaFold pipeline.
        
        Args:
            fold_input: The original input to AlphaFold.
            design_results: Results from the design process.
            rng_key: JAX random key.
            buckets: Optional bucket sizes for featurization.
            ref_max_modified_date: Optional reference date for chemical components.
            conformer_max_iterations: Optional iterations for conformer generation.
            
        Returns:
            Tuple of (model_result, final_feature_dict, new_fold_input).
        """
        logging.info("Running complete prediction with designed sequence...")
        
        # Get the final amino acid indices from the design results
        final_aa_indices = design_results["final_aa_indices"]
        
        # Convert to amino acid sequence using a proper mapping from data_constants
        try:
            from alphafold3.model import data_constants
            aa_types = data_constants.protein_restypes
        except (ImportError, AttributeError):
            # Fallback to standard alphabet if data_constants is not available
            aa_types = list("ACDEFGHIKLMNPQRSTVWY")
        
        # Map indices to amino acid sequences
        designed_sequence = ''.join([aa_types[idx] for idx in final_aa_indices])
        logging.info(f"Designed sequence ({len(designed_sequence)} aa): {designed_sequence[:50]}...")
        
        # Create a new fold_input with the designed sequence
        new_chains = []
        binder_chains = self.design_params["binder_chains"]
        
        # Create a mapping of chain ID to the new designed sequence
        binder_seq_by_chain = {}
        binder_start_idx = 0
        
        # First, build a mapping of chain ID to sequence length
        chain_lengths = {}
        for chain in fold_input.chains:
            chain_lengths[chain.id] = len(chain)
        
        # Build a mapping of binder chain ID to its designed sequence
        for binder_chain_id in binder_chains:
            if binder_chain_id in chain_lengths:
                chain_length = chain_lengths[binder_chain_id]
                # Extract the portion of the designed sequence for this chain
                binder_seq_by_chain[binder_chain_id] = designed_sequence[binder_start_idx:binder_start_idx + chain_length]
                binder_start_idx += chain_length
        
        # Create new chains with updated sequences where needed
        for chain in fold_input.chains:
            if isinstance(chain, folding_input.ProteinChain) and chain.id in binder_chains:
                # This is a binder chain - replace the sequence
                if chain.id in binder_seq_by_chain:
                    new_sequence = binder_seq_by_chain[chain.id]
                    # Create a new chain with the designed sequence
                    new_chain = folding_input.ProteinChain(
                        id=chain.id,
                        sequence=new_sequence,
                        ptms=chain.ptms,  # Keep the original PTMs
                        # Clear MSA and set templates to empty list to force re-search
                        unpaired_msa=None, 
                        paired_msa=None,
                        templates=[] # Use empty list instead of None
                    )
                    new_chains.append(new_chain)
                    logging.info(f"Replaced binder chain {chain.id} sequence and reset MSAs/templates.")
                else:
                    # Something went wrong - use the original chain
                    logging.warning(f"No designed sequence for binder chain {chain.id}, using original")
                    new_chains.append(chain)
            else:
                # This is a target chain or non-protein chain - keep it unchanged
                new_chains.append(chain)
        
        # Create a new Input object with the updated chains
        new_fold_input = folding_input.Input(
            name=fold_input.name + "_designed",
            chains=new_chains,
            rng_seeds=[fold_input.rng_seeds[0]],  # Use only the first seed
            bonded_atom_pairs=fold_input.bonded_atom_pairs,
            user_ccd=fold_input.user_ccd
        )
        
        logging.info("Created new input with designed binder sequence")
        
        # Now run a complete prediction with proper featurization
        try:
            from alphafold3.data import featurisation
            
            # Get the chemical components dictionary
            ccd = chemical_components.cached_ccd(user_ccd=new_fold_input.user_ccd)
            
            # Featurize the new input
            logging.info("Featurizing new input with designed sequence...")
            featurised_examples = featurisation.featurise_input(
                fold_input=new_fold_input,
                buckets=buckets,
                ccd=ccd,
                verbose=True,
                ref_max_modified_date=ref_max_modified_date,
                conformer_max_iterations=conformer_max_iterations,
                masking_config=None,  # No masking for final prediction
            )
            
            # Get the feature dictionary for the first (and only) seed
            final_feature_dict = featurised_examples[0]
            
            # Run standard prediction
            logging.info("Running inference on completely featurized designed sequence...")
            final_result = self.model_runner.run_inference(final_feature_dict, rng_key)
            
            logging.info("Complete prediction finished successfully")
            
            # Add designed sequence metadata
            if isinstance(final_result, dict) and "metadata" in final_result:
                final_result["metadata"]["designed_sequence"] = designed_sequence
                final_result["metadata"]["designed_binder_chains"] = binder_chains
                
            return final_result, final_feature_dict, new_fold_input
            
        except ImportError as e:
            logging.warning(f"Could not run complete prediction: {str(e)}")
            logging.warning("Falling back to simplified prediction method")
            
            # Fall back to the simpler approach
            final_result_simple, final_feature_dict_simple, new_fold_input_simple = self.run_final_prediction(fold_input, design_results, rng_key)
            return final_result_simple, final_feature_dict_simple, new_fold_input_simple


def design_binder(
    fold_input: folding_input.Input,
    feature_dict: features.BatchDict,
    model_runner: Any,
    ccd: chemical_components.Ccd,
    design_params: Dict[str, Any],
    rng_seed: int = 0,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    use_complete_prediction: bool = True,
) -> Tuple[Dict[str, Any], model.ModelResult, features.BatchDict, folding_input.Input]:
    """Design a binder protein using AlphaFold 3.
    
    Args:
        fold_input: The input to AlphaFold.
        feature_dict: The feature dictionary for the model.
        model_runner: ModelRunner instance.
        ccd: Chemical component dictionary.
        design_params: Dictionary of design parameters.
        rng_seed: Random seed for JAX.
        buckets: Optional bucket sizes for featurization of final prediction.
        ref_max_modified_date: Optional reference date for chemical components.
        conformer_max_iterations: Optional iterations for conformer generation.
        use_complete_prediction: Whether to use the comprehensive prediction method.
        
    Returns:
        Tuple of (design_results, final_model_result, final_feature_dict, new_fold_input).
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    
    # Initialize designer
    designer = BinderDesigner(model_runner, ccd, design_params)
    
    # Run design
    design_results, best_feature_dict = designer.design_binder(
        fold_input, feature_dict, rng_key
    )
    
    # Store best feature dict for final prediction
    design_results["best_feature_dict"] = best_feature_dict
    design_results["binder_indices"] = designer.design_params.get("binder_indices")
    design_results["final_seq_logits"] = design_results.get("final_seq_logits")
    
    # Run final prediction
    final_key, _ = jax.random.split(rng_key)
    
    # Choose which final prediction method to use
    if use_complete_prediction:
        final_model_result, final_feature_dict, new_fold_input = designer.run_complete_prediction(
            fold_input, 
            design_results, 
            final_key,
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations
        )
    else:
        final_model_result, final_feature_dict, new_fold_input = designer.run_final_prediction(
            fold_input, design_results, final_key
        )
    
    return design_results, final_model_result, final_feature_dict, new_fold_input 
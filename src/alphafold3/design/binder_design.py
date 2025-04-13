"""Binder design protocols for AlphaFold 3."""

import time
import functools
from typing import Dict, Any, Tuple, Sequence 
import datetime
import gc

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
from alphafold3.design import memory_utils
from alphafold3.constants import residue_names


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
        
        # Setup memory optimization parameters
        self.clear_memory_interval = design_params.get("clear_memory_interval", 0)
        
        logging.info(f"Initialized BinderDesigner with protocol: {self.protocol}")
        logging.info(f"Design parameters: {design_params}")
        if self.clear_memory_interval > 0:
            logging.info(f"Memory clearing enabled: clearing every {self.clear_memory_interval} steps")
    
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
        from alphafold3.common import memory_utils
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
        logging.info(f"Feature_dict_jax keys: {feature_dict_jax.keys()}")
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
            
            # Return only the loss breakdown as auxiliary data to prevent memory leaks
            return total_loss, loss_breakdown
        
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
            (loss, losses), grads = grad_fn(
                binder_seq_logits, feature_dict_jax, target_indices, binder_indices, 
                self.design_params, self.model_runner, step_key
            )
            
            # Update logits
            updates, opt_state = optimizer.update(grads, opt_state, binder_seq_logits)
            binder_seq_logits = optax.apply_updates(binder_seq_logits, updates)
            
            # Get loss value for logging without trying to convert JAX array to Python float inside JIT context
            step_loss_val = loss
            step_losses_val = losses
            
            # Log progress
            step_time = time.time() - step_start_time
            if step % 10 == 0 or step == steps - 1:
                probs = jax.nn.softmax(binder_seq_logits, axis=-1)
                aa_indices = jnp.argmax(probs, axis=-1)
                
                # Convert to amino acid sequence using proper residue names
                # Standard amino acid types in order
                aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
                
                # Convert JAX array indices to Python integers first
                aa_indices_py = [int(idx) for idx in aa_indices]
                current_seq = ''.join([aa_letters[idx] for idx in aa_indices_py])
                
                logging.info(f"Step {step}/{steps}: loss={safe_jax_to_float(step_loss_val):.4f}, time={step_time:.2f}s")
                safe_losses = safe_process_losses(step_losses_val)
                logging.info(f"Losses: {safe_losses}")
                
                trajectory["sequences"].append(current_seq)
            
            # Store in trajectory
            trajectory["loss"].append(step_loss_val)
            trajectory["losses"].append(step_losses_val)
            trajectory["step"].append(step)
            trajectory["time"].append(step_time)
            
            # Track best loss (comparing JAX arrays)
            if step_loss_val < best_loss:
                best_loss = step_loss_val
                best_logits = binder_seq_logits
                
                # Since we no longer get updated_feature_dict from grad_fn due to OOM prevention,
                # we need to regenerate it for the best state when needed
                # We'll only do this when we find a new best loss to limit computational overhead
                logging.info(f"Step {step}/{steps}: Regenerating feature dict for new best solution")
                best_feature_dict = binder_utils.update_features_from_logits(
                    jax.tree_map(lambda x: x.copy() if isinstance(x, (jnp.ndarray, np.ndarray)) else x, feature_dict_jax),
                    jnp.array(binder_indices, dtype=jnp.int32),
                    best_logits
                )
                
                logging.info(f"Step {step}/{steps}: New best loss {safe_jax_to_float(best_loss):.4f}")
            
            # Periodically clear memory if enabled
            if self.clear_memory_interval > 0 and step % self.clear_memory_interval == 0:
                logging.info(f"Step {step}/{steps}: Executing scheduled memory clearing")
                
                # Log memory usage before clearing
                mem_usage = memory_utils.get_current_memory_usage()
                if mem_usage:
                    logging.info(f"Before memory clearing: {mem_usage}")
                
                # Clear memory
                memory_utils.clear_memory(include_gpu=True, include_cpu=True)
                
                # Log memory usage after clearing
                mem_usage = memory_utils.get_current_memory_usage()
                if mem_usage:
                    logging.info(f"After memory clearing: {mem_usage}")
        
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
            "final_aa_indices": [int(aa) for aa in final_aa_indices],  # Convert to Python list of integers
        }
        
        logging.info(f"Gradient-based binder design completed in {design_results['design_time']:.2f}s")
        logging.info(f"Best loss: {safe_jax_to_float(best_loss):.4f}")
        
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
        logging.info("Starting boltz_forward_pass with feature_dict keys: " + str(list(feature_dict.keys())))
        
        # Validate key features are present
        required_keys = ['aatype', 'seq_mask', 'msa', 'msa_mask']
        missing_keys = [k for k in required_keys if k not in feature_dict]
        if missing_keys:
            logging.error(f"Missing required features for forward pass: {missing_keys}")
            raise ValueError(f"Missing required features for forward pass: {missing_keys}")
        
        # Log shapes of key features
        logging.info(f"aatype shape: {feature_dict['aatype'].shape}, seq_mask shape: {feature_dict['seq_mask'].shape}")
        logging.info(f"msa shape: {feature_dict['msa'].shape}, msa_mask shape: {feature_dict['msa_mask'].shape}")
        
        # Run the standard model inference but with a special mode parameter
        # that indicates this is a boltz design run (will stop_gradient at structure module)
        boltz_result = self.model_runner.run_inference(
            feature_dict, 
            rng_key,
            mode="boltz_design"  # Special mode to signal gradient stopping for BoltzDesign1
        )
        
        # Validate result structure
        if not isinstance(boltz_result, dict):
            logging.error(f"Expected dict result from run_inference, got: {type(boltz_result)}")
        
        # Check for essential keys in result
        result_keys = list(boltz_result.keys())
        logging.info(f"Boltz forward pass returned keys: {result_keys}")
        
        # Updated check: Look for top-level keys needed by the loss function
        expected_keys = ['distogram', 'predicted_lddt', 'full_pae'] 
        missing_result_keys = [k for k in expected_keys if k not in result_keys]
        if missing_result_keys:
            logging.warning(f"Potentially missing expected keys in boltz_result needed for loss: {missing_result_keys}")
        
        # Check distogram shape if present
        if 'distogram' in boltz_result:
            distogram_keys = list(boltz_result['distogram'].keys())
            logging.info(f"Distogram contains keys: {distogram_keys}")
            if 'contact_probs' in boltz_result['distogram']:
                contact_shape = boltz_result['distogram']['contact_probs'].shape
                logging.info(f"Contact_probs shape: {contact_shape}")
        
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
        from alphafold3.common import memory_utils
        logging.info("Starting Boltz-like binder design...")
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
            
            # --- Process sequence logits using JAX-compatible control flow ---
            
            # Stage 1: Exploration (T=1.0)
            def stage_0_fn(_):
                return jax.nn.softmax(curr_binder_logits / 1.0, axis=-1)
                
            # Stage 2: Transition
            def stage_1_fn(_):
                seq_probs_logits = jax.nn.softmax(curr_binder_logits, axis=-1)
                seq_probs_softmax = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
                return 0.5 * seq_probs_logits + 0.5 * seq_probs_softmax
                
            # Stage 3: Convergence (gradually decreasing temperature)
            def stage_2_fn(_):
                return jax.nn.softmax(curr_binder_logits / temp, axis=-1)
                
            # Stage 4: One-hot with straight-through estimator
            def stage_3_fn(_):
                seq_probs_soft = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
                seq_probs_hard = jax.nn.one_hot(jnp.argmax(seq_probs_soft, axis=-1), num_residue_types)
                return jax.lax.stop_gradient(seq_probs_hard - seq_probs_soft) + seq_probs_soft
                
            # Use JAX's switch statement equivalent
            seq_probs = jax.lax.switch(
                stage_idx,
                [stage_0_fn, stage_1_fn, stage_2_fn, stage_3_fn],
                None
            )
            
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
            
            # Log design_params_static for debugging
            logging.info(f"Design params static: {design_params_static} these are the weights that go into the loss function and should be printed as: Design weights: weights")
            jax.debug.print("Design weights: {p}", p=design_params_static)
            # Calculate BoltzDesign1 loss - use JAX arrays target_indices, binder_indices
            total_loss, loss_breakdown = binder_loss.calculate_boltz_binder_loss(
                partial_result=boltz_result, feature_dict=updated_feature_dict, target_indices=target_indices, binder_indices=binder_indices, 
                binder_seq_logits=curr_binder_logits, design_params=design_params_static
            )
            
            # Apply additional sequence shaping in one-hot stage using JAX-compatible control flow
            def apply_stage_3_entropy(args):
                loss, breakdown = args
                seq_probs_soft = jax.nn.softmax(curr_binder_logits / temp, axis=-1)
                seq_probs_hard = jax.nn.one_hot(jnp.argmax(seq_probs_soft, axis=-1), num_residue_types)
                seq_cross_entropy = -jnp.sum(seq_probs_hard * jax.nn.log_softmax(curr_binder_logits, axis=-1))
                seq_entropy_weight = design_params_static.get("weights", {}).get("seq_one_hot", 0.1)
                new_loss = loss + seq_entropy_weight * seq_cross_entropy
                new_breakdown = dict(breakdown)  # Create a copy to avoid mutation 
                new_breakdown["seq_one_hot"] = seq_cross_entropy
                return new_loss, new_breakdown
                
            def keep_as_is(args):
                loss, breakdown = args
                # Create a new dictionary with the same structure as apply_stage_3_entropy
                new_breakdown = dict(breakdown)
                new_breakdown["seq_one_hot"] = jnp.zeros((), dtype=jnp.float32)
                return loss, new_breakdown
                
            # Use JAX's conditional to apply one-hot entropy in stage 3 only
            total_loss, updated_breakdown = jax.lax.cond(
                stage_idx == 3,
                apply_stage_3_entropy,
                keep_as_is,
                (total_loss, loss_breakdown)
            )
            
            # Return only the loss breakdown as auxiliary data, not the full boltz_result or feature_dict
            # This is critical to avoid OOM errors during gradient computation by preventing large tensors 
            # from being included in the computation graph.
            return total_loss, updated_breakdown
        
        # Define the gradient function directly on the loss function
        grad_loss_fn = jax.value_and_grad(loss_fn_for_grad_boltz, argnums=0, has_aux=True)
        
        # JIT the gradient function, marking static arguments
        # Arguments: 0=logits, 1=stage, 2=temp, 3=key, 4=features, 5=tgt idx tuple, 6=bnd idx tuple, 7=weights_dict, 8=runner
        # Use the correctly defined static_argnums_for_jit
        grad_fn = jax.jit(grad_loss_fn, static_argnums=static_argnums_for_jit)
        #grad_fn = grad_loss_fn
        # Multi-stage optimization
        best_loss = float('inf')
        best_logits = None
        best_feature_dict = None
        current_step = 0
        stage_success = [False, False, False, False]  # Track success for each stage
        
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
            
            stage_had_successful_step = False
            
            for step in range(stage_steps):
                step_start_time = time.time()
                step_key, rng_key = jax.random.split(rng_key)
                
                # Check if memory should be cleared 
                if hasattr(self, 'clear_memory_interval') and self.clear_memory_interval > 0:
                    if current_step % self.clear_memory_interval == 0:
                        logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Clearing memory")
                        memory_utils.clear_mem()
                
                # Update temperature for stage 3 (gradually decreasing)
                if stage_idx == 2:
                    # Temperature decreases quadratically from 0.5 to 0.01
                    t_stage = (step + 1) / stage_steps
                    temp = 0.01 + (0.5 - 0.01) * (1.0 - t_stage)**2
                
                # Call the JITted grad_fn with all arguments explicitly
                # Passing the tuples for static indices arguments and the filtered feature dict
                logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: About to call grad_fn")
                
                # Initialize step results with defaults
                step_loss_val = float('nan')
                step_losses_val = {"error": "step_failed"}
                step_success = False
                
                try:
                    # Calling grad_fn with the boltz approach returns only (loss, losses) as aux data,
                    # not the full (updated_breakdown, boltz_result, updated_feature_dict) which caused OOM
                    (loss, losses), grads = grad_fn(
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
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: grad_fn call successful")
                    
                    # Check gradients for NaN/Inf
                    grad_has_nan = jnp.any(jnp.isnan(jax.tree_map(lambda x: jnp.sum(x), grads)))
                    grad_has_inf = jnp.any(jnp.isinf(jax.tree_map(lambda x: jnp.sum(x), grads)))
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Gradient check - has_nan: {grad_has_nan}, has_inf: {grad_has_inf}")
                    
                    # Skip update if gradients contain NaN/Inf
                    if grad_has_nan or grad_has_inf:
                        logging.warning(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Skipping update due to NaN/Inf in gradients")
                        # Get loss value for logging without trying to convert JAX array to Python float inside JIT context
                        step_loss_val = loss
                        step_losses_val = losses
                        continue
                    
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: About to update optimizer")
                    # Update logits
                    updates, opt_state = optimizer.update(grads, opt_state, binder_seq_logits)
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Optimizer update successful")
                    
                    # Check updates for NaN/Inf
                    updates_has_nan = jnp.any(jnp.isnan(jax.tree_map(lambda x: jnp.sum(x), updates)))
                    updates_has_inf = jnp.any(jnp.isinf(jax.tree_map(lambda x: jnp.sum(x), updates)))
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Updates check - has_nan: {updates_has_nan}, has_inf: {updates_has_inf}")
                    
                    # Skip applying updates if they contain NaN/Inf
                    if updates_has_nan or updates_has_inf:
                        logging.warning(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Skipping updates due to NaN/Inf")
                        # Get loss value for logging without trying to convert JAX array to Python float inside JIT context
                        step_loss_val = loss
                        step_losses_val = losses
                        continue
                    
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: About to apply updates")
                    binder_seq_logits = optax.apply_updates(binder_seq_logits, updates)
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Applied updates successfully")
                    
                    # Check updated logits for NaN/Inf
                    logits_has_nan = jnp.any(jnp.isnan(binder_seq_logits))
                    logits_has_inf = jnp.any(jnp.isinf(binder_seq_logits))
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Logits check - has_nan: {logits_has_nan}, has_inf: {logits_has_inf}")
                    
                    # Mark this step and stage as successful since we got this far
                    step_success = True
                    stage_had_successful_step = True
                    
                    # Get loss value for logging without trying to convert JAX array to Python float inside JIT context
                    step_loss_val = loss
                    step_losses_val = losses
                    
                    # Track best loss (comparing JAX arrays)
                    if step_loss_val < best_loss:
                        best_loss = step_loss_val
                        best_logits = binder_seq_logits
                        
                        # Since we no longer get updated_feature_dict from grad_fn due to OOM prevention,
                        # we need to regenerate it for the best state when needed
                        # We'll only do this when we find a new best loss to limit computational overhead
                        logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Regenerating feature dict for new best solution")
                        best_feature_dict = binder_utils.update_features_from_logits(
                            jax.tree_map(lambda x: x.copy() if isinstance(x, (jnp.ndarray, np.ndarray)) else x, feature_dict_template),
                            jnp.array(binder_indices_static_tuple, dtype=jnp.int32),
                            best_logits
                        )
                        
                        logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: New best loss {safe_jax_to_float(best_loss):.4f}")
                    
                    # Periodically clear memory if enabled
                    if self.clear_memory_interval > 0 and step % self.clear_memory_interval == 0:
                        logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: Executing scheduled memory clearing")
                        
                        # Log memory usage before clearing
                        mem_usage = memory_utils.get_current_memory_usage()
                        if mem_usage:
                            logging.info(f"Before memory clearing: {mem_usage}")
                        
                        # Clear memory
                        memory_utils.clear_memory(include_gpu=True, include_cpu=True)
                        
                        # Log memory usage after clearing
                        mem_usage = memory_utils.get_current_memory_usage()
                        if mem_usage:
                            logging.info(f"After memory clearing: {mem_usage}")
                
                except Exception as e:
                    logging.error(f"❌ Error in Stage {stage_idx+1}, Step {step}/{stage_steps}: {e}", exc_info=True)
                    # Don't immediately re-raise, try to continue with next step
                
                # Log progress
                step_time = time.time() - step_start_time
                if step % 10 == 0 or step == stage_steps - 1:
                    probs = jax.nn.softmax(binder_seq_logits / temp, axis=-1)
                    aa_indices = jnp.argmax(probs, axis=-1)
                    
                    # Convert to amino acid sequence using proper residue names
                    # Standard amino acid types in order
                    aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
                    
                    # Convert JAX array indices to Python integers first
                    aa_indices_py = [int(idx) for idx in aa_indices]
                    current_seq = ''.join([aa_letters[idx] for idx in aa_indices_py])
                    
                    logging.info(f"Stage {stage_idx+1}, Step {step}/{stage_steps}: loss={safe_jax_to_float(step_loss_val):.4f}, time={step_time:.2f}s")
                    safe_losses = safe_process_losses(step_losses_val)
                    logging.info(f"Losses: {safe_losses}")
                    
                    if step_success:
                        trajectory["sequences"].append(current_seq)
                
                # Store in trajectory
                if step_success:
                    trajectory["loss"].append(step_loss_val)
                    trajectory["losses"].append(step_losses_val)
                    trajectory["step"].append(current_step)
                    trajectory["time"].append(step_time)
                    trajectory["stage"].append(stage_idx)
                
                current_step += 1
            
            # Update stage success tracker
            stage_success[stage_idx] = stage_had_successful_step
            if not stage_had_successful_step:
                logging.warning(f"Stage {stage_idx+1} had no successful steps")

        # Final sequence - in the one-hot stage
        temp = 0.01  # Very low temperature for final output
        
        # Verify we have valid results
        if best_logits is None or best_feature_dict is None:
            logging.error("❌ Binder design optimization failed: No valid solution found")
            # Return a minimally valid result with error flag
            design_results = {
                "protocol": "binder_boltz",
                "target_indices": target_indices,
                "binder_indices": binder_indices,
                "best_loss": float('inf'),
                "trajectory": trajectory,
                "design_time": time.time() - design_start_time,
                "error": "No valid solution found during optimization",
                "final_aa_indices": np.zeros_like(binder_indices, dtype=np.int32),
                "success": False,
                "stage_success": stage_success
            }
            return design_results, feature_dict_template
            
        final_probs = jax.nn.softmax(best_logits / temp, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)
        
        # At the end of the method, before returning results:
        logging.info("======================================")
        logging.info("Completing binder design optimization loop")
        logging.info(f"Processed {current_step} total steps across {len(stages)} stages")
        logging.info(f"Best loss achieved: {safe_jax_to_float(best_loss):.6f}")
        logging.info(f"Stage success summary: {stage_success}")
        logging.info(f"Designed sequence length: {len(final_aa_indices)}")
        
        # Get amino acid sequence
        # Create a mapping from index to one-letter codes
        restype_idx_to_letter = {i: aa for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
        # Convert JAX array elements to Python integers first
        final_aa_indices_py = [int(aa) for aa in final_aa_indices]
        final_sequence = ''.join([restype_idx_to_letter[aa] for aa in final_aa_indices_py])
        
        logging.info(f"Designed sequence: {final_sequence}")
        logging.info("======================================")
        
        # Prepare results
        design_results = {
            "protocol": "binder_boltz",
            "target_indices": target_indices,
            "binder_indices": binder_indices,
            "best_loss": best_loss,
            "trajectory": trajectory,
            "design_time": time.time() - design_start_time,
            "final_aa_indices": [int(aa) for aa in final_aa_indices],  # Convert to Python list of integers
            "final_sequence": final_sequence,
            "success": any(stage_success),
            "stage_success": stage_success
        }
        
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
        
        # Check if design was successful
        if not design_results.get("success", True):  # Default to True for backward compatibility
            logging.error("Cannot run final prediction: Design process was unsuccessful")
            # Return empty/placeholder results
            empty_result = {"error": "Failed design process", "status": "error"}
            empty_feature_dict = {}
            return empty_result, empty_feature_dict, fold_input
        
        try:
            # Get the final amino acid indices from the design results
            final_aa_indices = design_results["final_aa_indices"]
            
            # Convert to amino acid sequence using residue names constants
            from alphafold3.constants import residue_names
            # Standard amino acid types in order
            aa_types = 'ACDEFGHIKLMNPQRSTVWY'
            
            # Map indices to amino acid sequences
            designed_sequence = ''.join([aa_types[int(idx)] for idx in final_aa_indices])
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
            try:
                final_feature_dict = binder_utils.update_features_from_logits(
                    design_results.get("best_feature_dict", {}).copy(),
                    design_results["binder_indices"],
                    design_results["final_seq_logits"]
                )
            except Exception as e:
                logging.error(f"Error updating features from logits: {e}")
                # If we fail to update from the design results, try to use the original dictionary
                if "best_feature_dict" in design_results:
                    final_feature_dict = design_results["best_feature_dict"].copy()
                else:
                    # Last resort, use an empty dict
                    final_feature_dict = {}
            
            # Run standard prediction
            logging.info("Running inference on designed sequence (Simplified Method)...")
            final_result = self.model_runner.run_inference(final_feature_dict, rng_key)
            
            logging.info("Final prediction complete (Simplified Method)")
            
            # Add designed sequence to the result metadata
            if isinstance(final_result, dict) and "metadata" in final_result:
                final_result["metadata"]["designed_sequence"] = designed_sequence
            
            return final_result, final_feature_dict, new_fold_input
            
        except Exception as e:
            logging.error(f"Error in final prediction: {e}", exc_info=True)
            # Return empty/placeholder results
            empty_result = {"error": str(e), "status": "error"}
            empty_feature_dict = {}
            return empty_result, empty_feature_dict, fold_input

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
        
        # Convert to amino acid sequence using residue names constants
        from alphafold3.constants import residue_names
        # Standard amino acid types in order
        aa_types = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Map indices to amino acid sequences
        designed_sequence = ''.join([aa_types[int(idx)] for idx in final_aa_indices])
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

def safe_jax_to_float(x):
    """Safely convert JAX array to float for logging, handling potential errors."""
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float('nan')

def safe_process_losses(losses):
    """Process dictionary of JAX losses to Python values for logging."""
    if not isinstance(losses, dict):
        return {"loss": safe_jax_to_float(losses)}
    
    return {k: safe_jax_to_float(v) if hasattr(v, "item") or not isinstance(v, dict) 
            else safe_process_losses(v) for k, v in losses.items()} 
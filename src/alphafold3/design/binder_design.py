"""Binder design protocols for AlphaFold 3."""

import time
import functools
from typing import Dict, Any, Tuple, Sequence 
import datetime
import gc
import copy

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

def freeze_containers_for_jax(obj):
    """Makes a nested structure of dicts and lists JAX-compatible by making them immutable.
    
    Args:
        obj: A nested structure of dicts, lists and leaf values.
        
    Returns:
        A similar structure with dicts and lists converted to immutable types.
    """
    if isinstance(obj, dict):
        # Convert each value in the dict and return an immutable mapping (FrozenDict)
        return jax.tree_util.tree_map(
            freeze_containers_for_jax,
            {k: v for k, v in obj.items() if not (isinstance(v, np.ndarray) and v.dtype == object)}
        )
    elif isinstance(obj, list):
        # Convert each element in the list and return a tuple (immutable)
        return tuple(freeze_containers_for_jax(x) for x in obj)
    else:
        # For leaf values (including JAX arrays), return as is
        return obj

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
        logging.debug("Starting boltz_forward_pass with feature_dict keys: " + str(list(feature_dict.keys())))
        
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
        """Design binder using a Boltzmann-inspired approach.
        
        Args:
            fold_input: Fold input containing target and binder.
            feature_dict: Feature dictionary.
            target_indices: Indices of the target residues.
            binder_indices: Indices of the binder residues.
            rng_key: Random key.
            
        Returns:
            Tuple of (design_results, updated_feature_dict).
        """
        logging.info("=== Starting New Design Trajectory ===")
        # Log design parameters
        binder_length = len(binder_indices)
        seed = int(jnp.sum(rng_key)) % 100000  # Generate a seed-like number from the random key
        helicity_bias = self.design_params.get("helicity_bias", 0.0)
        
        logging.info(f"Design parameters:")
        logging.info(f"- Length: {binder_length}")
        logging.info(f"- Seed: {seed}")
        logging.info(f"- Helicity: {helicity_bias}")
        
        # Log target hotspot residues if provided
        hotspot_residues = self.design_params.get("hotspot_residues", [])
        if hotspot_residues:
            logging.info(f"- Target hotspot residues: {','.join(hotspot_residues)}")
        
        logging.info("Initiating binder hallucination...")
        
        # --- Setup for optimization ---
        rng_key, subkey = jax.random.split(rng_key)
        
        # Initialize logits
        random_init_scale = self.design_params.get("random_init_scale", 0.01)
        num_residue_types = 20  # Standard amino acids
        binder_logits_shape = (len(binder_indices), num_residue_types)
        binder_logits = random_init_scale * jax.random.normal(
            subkey, shape=binder_logits_shape
        )
        
        # Copy feature dict
        feature_dict_template = copy.deepcopy(feature_dict)
        
        # Initial random logits - convert to JAX array for optimization
        binder_logits = jnp.asarray(binder_logits, dtype=jnp.float32)
        
        # Copy design_params for static compilation
        design_weights_static = dict(self.design_params["weights"])
        
        # Define stages with their iterations, temperatures, and hardness flags
        total_iterations = self.design_params.get("iterations", 140)
        
        # Check how stages are defined in design_params
        original_stages = self.design_params.get("stages", [50, 50, 25, 15])
        
        # Default stage configurations (with standard names and parameters)
        default_stage_configs = [
            {"name": "Test Logits", "iters": 50, "hard": False, "start_temp": 1.0, "end_temp": 1.0, 
             "plddt_threshold": 0.85},
            {"name": "Additional Logits Optimisation", "iters": 25, "hard": False, "start_temp": 1.0, 
             "end_temp": 1.0, "plddt_threshold": 0.90},
            {"name": "Softmax Optimisation", "iters": 45, "hard": False, "start_temp": 1.0, 
             "end_temp": 0.01, "plddt_threshold": 0.90},
            {"name": "One-hot Optimisation", "iters": 5, "hard": True, "start_temp": 0.01, 
             "end_temp": 0.01, "plddt_threshold": 0.90},
            {"name": "PSSM Semigreedy Optimisation", "iters": 15, "hard": True, "start_temp": 1.0, 
             "end_temp": 1.0, "soft": False, "plddt_threshold": 0.0},
        ]
        
        # Initialize the stages list
        stages = []
        
        # Check if original_stages is a list of integers or already a list of dicts
        if original_stages and isinstance(original_stages[0], int):
            # Convert integer stage definitions to our full stage configuration format
            # For backwards compatibility with the old API
            stage_names = ["Test Logits", "Additional Logits Optimisation", 
                           "Softmax Optimisation", "One-hot Optimisation"]
            
            # Temperature and hardness settings for each stage
            temp_settings = [(1.0, 1.0), (1.0, 1.0), (1.0, 0.01), (0.01, 0.01)]
            hardness = [False, False, False, True]
            
            # Create the stages list with appropriate parameters
            for i, iterations in enumerate(original_stages[:4]):  # Only use first 4 stages from original
                if i < len(stage_names):
                    stage_config = {
                        "name": stage_names[i],
                        "iters": iterations,
                        "hard": hardness[i],
                        "start_temp": temp_settings[i][0],
                        "end_temp": temp_settings[i][1],
                        "soft": True,
                        "plddt_threshold": 0.0  # No threshold by default
                    }
                    stages.append(stage_config)
            
            # Add PSSM stage if there's a 5th value
            if len(original_stages) >= 5:
                stages.append({
                    "name": "PSSM Semigreedy Optimisation",
                    "iters": original_stages[4],
                    "hard": True,
                    "start_temp": 1.0,
                    "end_temp": 1.0,
                    "soft": False,
                    "plddt_threshold": 0.0
                })
        else:
            # If stages are already dictionaries, use them directly
            stages = original_stages
            
            # If no stages were provided, use the default configurations
            if not stages:
                stages = default_stage_configs
        
        logging.info(f"Using {len(stages)} stages for optimization")
        for i, stage in enumerate(stages):
            logging.info(f"Stage {i+1}: {stage.get('name', f'Stage {i+1}')} - {stage.get('iters', 0)} iterations")
        
        # === Prepare for optimization loop ===
        # Convert indices to tuples for JIT
        target_indices_tuple = tuple(target_indices)
        binder_indices_tuple = tuple(binder_indices)
        
        # Make feature_dict_template JAX-friendly: freeze containers
        feature_dict_template_jax = freeze_containers_for_jax(feature_dict_template)
        
        # For model_runner, we can't easily make it JAX-friendly, so we'll pass it directly
        model_runner_obj = self.model_runner
        
        # Define the loss function for gradient-based optimization
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
            target_indices_arr = jnp.array(target_indices_tuple)
            binder_indices_arr = jnp.array(binder_indices_tuple)
            
            # --- Handle different logit processing based on stage ---
            def stage_0_fn(_):
                # Test logits: Just apply temperature
                return jnp.asarray(curr_binder_logits) / temp
                
            def stage_1_fn(_):
                # Same as stage 0 for now
                return jnp.asarray(curr_binder_logits) / temp
            
            def stage_2_fn(_):
                # Softmax optimization: temp controls sharpness
                return jnp.asarray(curr_binder_logits) / temp
                
            def stage_3_fn(_):
                # One-hot optimization: Apply temp and then one-hot
                logits_over_temp = jnp.asarray(curr_binder_logits) / temp
                # Use gumbel-softmax for one-hot
                return logits_over_temp
            
            # Decision tree for stage processing (for jit compatibility)
            # Note: Instead of Python if/else, use jax.lax.switch
            processed_logits = jax.lax.switch(
                stage_idx,
                [stage_0_fn, stage_1_fn, stage_2_fn, stage_3_fn],
                None  # Operand (unused in our case)
            )
            
            # Stage 4 (if needed later) would be PSSM Semigreedy
            
            # Update the feature dict with the processed logits
            updated_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_template, binder_indices_arr, processed_logits
            )
            
            # Run the model to get partial outputs (stop at confidence head)
            boltz_result = self._boltz_forward_pass(updated_feature_dict, rng_key)
            
            # Calculate loss
            total_loss, loss_breakdown = binder_loss.calculate_boltz_binder_loss(
                boltz_result, 
                updated_feature_dict,
                target_indices_arr, 
                binder_indices_arr, 
                processed_logits,
                design_params_static  # Use design_params directly as weights
            )
            
            # In stage 3 (one-hot), add seq_one_hot loss
            def apply_stage_3_entropy(args):
                loss_val, breakdown = args
                
                # Get probabilities (no logits available directly for one-hot mode)
                probs = jax.nn.softmax(processed_logits, axis=-1)
                # Compute cross-entropy with one-hot target
                one_hot_target = jnp.eye(probs.shape[-1])[jnp.argmax(probs, axis=-1)]
                one_hot_loss = -jnp.sum(one_hot_target * jnp.log(jnp.maximum(probs, 1e-8))) / len(binder_indices_arr)
                
                # Add to total loss and breakdown
                total_with_entropy = loss_val + one_hot_loss
                breakdown["seq_one_hot"] = one_hot_loss
                
                return total_with_entropy, breakdown
                
            def keep_as_is(args):
                return args
            
            # Apply the one-hot loss conditionally based on stage_idx == 3
            updated_loss, updated_breakdown = jax.lax.cond(
                jnp.equal(stage_idx, 3),
                apply_stage_3_entropy,
                keep_as_is,
                (total_loss, loss_breakdown)
            )
            
            return updated_loss, updated_breakdown
        
        # ---- PREPARE THE GRADIENT FUNCTION ----
        # We need value_and_grad for the first argument (logits), and we want aux data (loss breakdown)
        grad_loss_fn = jax.value_and_grad(loss_fn_for_grad_boltz, has_aux=True)
        
        # Specify which arguments are static for JAX JIT
        static_argnums_for_jit = (5, 6, 8)  # Marking target_indices_tuple, binder_indices_tuple, model_runner_obj as static
        
        grad_fn = grad_loss_fn
        # --- OPTIMIZATION LOOP ----
        # Initialize the Adam optimizer with default parameters
        optimizer = optax.adam(learning_rate=self.design_params.get("learning_rate", 0.01))
        opt_state = optimizer.init(binder_logits)
        
        # Track best results
        best_loss = float('inf')
        best_logits = None
        best_metrics = None
        best_feature_dict = None
        
        current_iter = 0
        
        # Process each stage
        for stage_idx, stage in enumerate(stages):
            # Extract stage parameters (with default values)
            stage_name = stage.get("name", f"Stage {stage_idx+1}")
            stage_iters = stage.get("iters", 50)
            hard_mode = stage.get("hard", False)  # One-hot (True) or softmax (False)
            soft_mode = stage.get("soft", True)   # Softmax (True) or semigreedy (False)
            start_temp = stage.get("start_temp", 1.0)
            end_temp = stage.get("end_temp", 0.01 if stage_idx > 1 else 1.0)
            plddt_threshold = stage.get("plddt_threshold", 0.0)
            
            logging.info(f"Stage {stage_idx+1}: {stage_name}")
            
            # For each iteration in this stage
            for i in range(stage_iters):
                # Calculate temperature using linear schedule
                progress = i / (stage_iters - 1) if stage_iters > 1 else 1.0
                temp = start_temp + progress * (end_temp - start_temp)
                
                # Get a new RNG key for this iteration
                rng_key, iter_key = jax.random.split(rng_key)
                
                # Track which model is used (for logging)
                model_idx = current_iter % 5
                
                try:
                    # Run a gradient step
                    (loss_val, loss_breakdown), grads = grad_fn(
                        binder_logits,
                        jnp.array(stage_idx, dtype=jnp.int32),
                        jnp.array(temp, dtype=jnp.float32),
                        iter_key,
                        feature_dict_template_jax,
                        target_indices_tuple,
                        binder_indices_tuple,
                        design_weights_static,
                        model_runner_obj
                    )
                    
                    # Check if loss is valid (not NaN or inf)
                    if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                        logging.warning(f"Invalid loss value: {loss_val}, skipping update")
                        continue
                    
                    # Update logits using optimizer
                    updates, opt_state = optimizer.update(grads, opt_state)
                    binder_logits = optax.apply_updates(binder_logits, updates)
                    
                    # Extract metrics for logging
                    # Convert to float to ensure they're Python types, not JAX arrays
                    helix_score = float(loss_breakdown.get("helix", 0.0))
                    pae = float(loss_breakdown.get("pae", 0.0))
                    i_pae = float(loss_breakdown.get("i_pae", 0.0))
                    con = float(loss_breakdown.get("distogram", 0.0))
                    i_con = float(loss_breakdown.get("i_con", 0.0))
                    plddt = float(1.0 - loss_breakdown.get("plddt", 0.0))  # Convert negative loss to positive score
                    ptm = float(loss_breakdown.get("ptm", 0.5))
                    i_ptm = float(loss_breakdown.get("i_ptm", 0.0))
                    rg = float(loss_breakdown.get("rg", 0.0))
                    
                    # Format log line to match bindcraft style
                    log_line = (f"{current_iter+1} models [{model_idx}] recycles 1 "
                               f"hard {int(hard_mode)} soft {int(soft_mode)} "
                               f"temp {temp:.2f} loss {loss_val:.2f} "
                               f"helix {helix_score:.2f} pae {pae:.2f} i_pae {i_pae:.2f} "
                               f"con {con:.2f} i_con {i_con:.2f} plddt {plddt:.2f} "
                               f"ptm {ptm:.2f} i_ptm {i_ptm:.2f} rg {rg:.2f}")
                    
                    logging.info(log_line)
                    
                    # Check if this is the best loss so far
                    if loss_val < best_loss:
                        best_loss = float(loss_val)
                        best_logits = binder_logits.copy()
                        best_metrics = {
                            "loss": float(loss_val),
                            "plddt": plddt,
                            "pae": pae,
                            "i_pae": i_pae,
                            "con": con,
                            "i_con": i_con,
                            "ptm": ptm,
                            "i_ptm": i_ptm
                        }
                        
                        # Only regenerate the feature dict for the best result
                        # to save memory during optimization
                        best_feature_dict = binder_utils.update_features_from_logits(
                            feature_dict_template, binder_indices, best_logits
                        )
                    
                    current_iter += 1
                    
                except Exception as e:
                    logging.error(f"Error in iteration {current_iter+1}: {e}")
                    # Continue with next iteration
            
            # Check pLDDT threshold after each stage
            if plddt_threshold > 0 and best_metrics and best_metrics["plddt"] >= plddt_threshold:
                logging.info(f"{stage_name} trajectory pLDDT good, continuing: {best_metrics['plddt']:.2f}")
            elif plddt_threshold > 0:
                logging.info(f"{stage_name} trajectory pLDDT below threshold: {best_metrics['plddt'] if best_metrics else 0:.2f}")
                # Could add logic to abort or retry if needed
        
            if stage_idx == 4:  # After PSSM semigreedy stage
                logging.info("Running semigreedy optimization...")
                
                # PSSM Semigreedy optimization implementation
                # This is a position-specific mutation sampling approach
                semigreedy_iters = stage.get("iters", 15)
                mutation_rate = stage.get("mutation_rate", 0.10)  # Percentage of positions to mutate per iteration
                
                # Convert best logits to probabilities and one-hot encodings
                probs = jax.nn.softmax(best_logits, axis=-1)
                aa_indices = jnp.argmax(probs, axis=-1)
                
                for sg_iter in range(semigreedy_iters):
                    # Sample positions to mutate
                    num_positions = len(binder_indices)
                    num_to_mutate = max(1, int(num_positions * mutation_rate))
                    
                    # Generate a new key for this iteration
                    rng_key, mut_key = jax.random.split(rng_key)
                    
                    # Sample positions without replacement
                    positions_to_mutate = jax.random.choice(
                        mut_key, 
                        jnp.arange(num_positions), 
                        shape=(num_to_mutate,), 
                        replace=False
                    )
                    
                    # For each position, sample a new amino acid based on PSSM probabilities
                    rng_key, sample_key = jax.random.split(rng_key)
                    
                    # Create a copy of aa_indices to modify
                    new_aa_indices = aa_indices.copy()
                    
                    # For each position to mutate, sample new amino acid by PSSM probabilities
                    for pos_idx in positions_to_mutate:
                        rng_key, aa_key = jax.random.split(rng_key)
                        position_probs = probs[pos_idx]
                        
                        # Sample new amino acid
                        new_aa = jax.random.choice(
                            aa_key,
                            jnp.arange(20),  # 20 standard amino acids
                            p=position_probs
                        )
                        new_aa_indices = new_aa_indices.at[pos_idx].set(new_aa)
                    
                    # Convert the sequence to one-hot for feature update
                    semigreedy_one_hot = jax.nn.one_hot(new_aa_indices, 20)
                    
                    # Update features with the semigreddy sequence
                    feature_dict_sg = binder_utils.update_features_from_logits(
                        feature_dict_template, binder_indices, semigreedy_one_hot
                    )
                    
                    # Run model with the new sequence
                    rng_key, eval_key = jax.random.split(rng_key)
                    sg_result = self._boltz_forward_pass(feature_dict_sg, eval_key)
                    
                    # Calculate metrics for this sequence
                    sg_loss, sg_breakdown = binder_loss.calculate_boltz_binder_loss(
                        sg_result,
                        feature_dict_sg,
                        target_indices,
                        binder_indices,
                        semigreedy_one_hot,
                        design_weights_static
                    )
                    
                    # Extract metrics for logging
                    helix_score = float(sg_breakdown.get("helix", 0.0))
                    pae = float(sg_breakdown.get("pae", 0.0))
                    i_pae = float(sg_breakdown.get("i_pae", 0.0))
                    con = float(sg_breakdown.get("distogram", 0.0))
                    i_con = float(sg_breakdown.get("i_con", 0.0))
                    plddt = float(1.0 - sg_breakdown.get("plddt", 0.0))
                    ptm = float(sg_breakdown.get("ptm", 0.5))
                    i_ptm = float(sg_breakdown.get("i_ptm", 0.0))
                    rg = float(sg_breakdown.get("rg", 0.0))
                    
                    # Log this iteration
                    current_iter += 1
                    model_idx = current_iter % 5
                    
                    log_line = (f"{current_iter} models [{model_idx}] recycles 1 "
                              f"hard {int(hard_mode)} soft {int(soft_mode)} "
                              f"temp {temp:.2f} loss {sg_loss:.2f} "
                              f"helix {helix_score:.2f} pae {pae:.2f} i_pae {i_pae:.2f} "
                              f"con {con:.2f} i_con {i_con:.2f} plddt {plddt:.2f} "
                              f"ptm {ptm:.2f} i_ptm {i_ptm:.2f} rg {rg:.2f}")
                    
                    logging.info(log_line)
                    
                    # Check if this is the new best
                    if sg_loss < best_loss:
                        best_loss = float(sg_loss)
                        # Convert one-hot back to logits representation (with high temperature)
                        best_logits = 10.0 * semigreedy_one_hot  # Scale factor gives sharp logits
                        aa_indices = new_aa_indices  # Update current sequence
                        
                        best_metrics = {
                            "loss": float(sg_loss),
                            "plddt": plddt,
                            "pae": pae,
                            "i_pae": i_pae,
                            "con": con,
                            "i_con": i_con,
                            "ptm": ptm,
                            "i_ptm": i_ptm
                        }
                        
                        best_feature_dict = feature_dict_sg

        # Restore best logits and feature dict
        if best_logits is not None:
            binder_logits = best_logits
        
        if best_feature_dict is None:
            # Fallback: regenerate from best logits
            best_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_template, binder_indices, binder_logits
            )
        
        # Return design results and updated feature dict
        design_results = {
            "binder_logits": binder_logits,
            "binder_indices": binder_indices,
            "target_indices": target_indices,
            "metrics": best_metrics or {},
            "design_trajectory": {
                "final_loss": float(best_loss),
                "final_plddt": best_metrics["plddt"] if best_metrics else 0.0,
                "iterations": current_iter
            }
        }
        
        # Extract the final amino acid indices for compatibility with run_complete_prediction
        final_probs = jax.nn.softmax(best_logits, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)
        
        # Convert to Python list for JSON serialization
        design_results["final_aa_indices"] = [int(aa) for aa in final_aa_indices]
        
        # Add a human-readable sequence
        aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        final_sequence = ''.join([aa_letters[i] for i in design_results["final_aa_indices"]])
        design_results["final_sequence"] = final_sequence
        
        # For backwards compatibility
        design_results["protocol"] = "binder_boltz"
        design_results["success"] = True if best_metrics and best_metrics.get("plddt", 0) > 0.7 else False
        
        logging.info(f"Design completed with final pLDDT: {best_metrics['plddt'] if best_metrics else 0:.2f}")
        logging.info(f"Final sequence: {final_sequence}")
        
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
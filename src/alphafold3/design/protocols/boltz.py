"""Boltzmann-inspired binder design protocol."""

import time
import logging
import copy
from typing import Dict, Any, Tuple, Sequence, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import matplotlib.pyplot as plt
import dataclasses

from alphafold3.common import folding_input
from alphafold3.common import memory_utils
from alphafold3.constants import chemical_components
from alphafold3.model import model
from alphafold3.model import features
from alphafold3.design import binder_utils
from alphafold3.design.losses import calculate_boltzdesign_binder_loss
from alphafold3.design.utils import freeze_containers_for_jax, safe_jax_to_float, safe_process_losses
from .base import BinderProtocol
from alphafold3.design.sequence_utils import soft_seq_af3
from alphafold3.design.config import BoltzDesignConfig, ALPHABET_SIZE

# Helper to plot binder_logits softmax probabilities per step
def _plot_logits(logits, step, prefix="boltz"):
    import jax
    probs = jax.nn.softmax(logits, axis=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    c = ax.imshow(probs.T, aspect='auto', origin='lower')
    ax.set_xlabel('Binder Position')
    ax.set_ylabel('AA Index')
    ax.set_title(f'{prefix} Step {step} Softmax Probabilities')
    plt.colorbar(c, ax=ax)
    os.makedirs(f"{prefix}_plots", exist_ok=True)
    fig.savefig(os.path.join(f"{prefix}_plots", f"{prefix}_step_{step}.png"))
    plt.close(fig)

class BoltzProtocol(BinderProtocol):
    """Implements the Boltzmann-inspired (BoltzDesign1) binder design protocol."""

    def _boltz_forward_pass(self, feature_dict, rng_key, stop_gradient=True):
        """Modified forward pass for BoltzDesign1 approach.

        Runs the model with a special mode to potentially stop gradients before
        the structure module, depending on the model runner implementation.

        Args:
            feature_dict: The feature dictionary for the model.
            rng_key: JAX random key.
            stop_gradient: (Currently unused, logic is inside model_runner.run_inference)
                Intended to control gradient flow.

        Returns:
            Partial or full results from the forward pass.
        """
        logging.debug("Starting boltz_forward_pass...")

        # Validate key features are present
        required_keys = ['aatype', 'seq_mask'] # Simplified check
        missing_keys = [k for k in required_keys if k not in feature_dict]
        if missing_keys:
            logging.error(f"Missing required features for forward pass: {missing_keys}")
            raise ValueError(f"Missing required features for forward pass: {missing_keys}")

        # Log shapes of key features
        #logging.debug(f"aatype shape: {feature_dict['aatype'].shape}, seq_mask shape: {feature_dict['seq_mask'].shape}")
        #if 'msa' in feature_dict:
        #     logging.debug(f"msa shape: {feature_dict['msa'].shape}, msa_mask shape: {feature_dict.get('msa_mask', 'N/A')}")

        # Run inference with 'boltz_design' mode (ModelRunner handles recycle=0)
        # It's assumed that the ModelRunner's call to the underlying model
        # correctly recomputes Confidence head outputs based on the
        # potentially updated trunk features passed in feature_dict, rather
        # than caching stale confidence outputs.
        boltz_result = self.model_runner.run_inference(
            feature_dict,
            rng_key,
            mode="boltz_design"
            # No need to pass num_recycles_override here, mode handles it
        )

        # Basic validation of result structure
        if not isinstance(boltz_result, dict):
            logging.error(f"Expected dict result from run_inference, got: {type(boltz_result)}")
            # Attempt to recover if possible, or raise
            if hasattr(boltz_result, '__dict__'):
                 boltz_result = dict(boltz_result.__dict__) # Basic conversion attempt
            else:
                 raise TypeError(f"Unexpected result type from model runner: {type(boltz_result)}")

        result_keys = list(boltz_result.keys())
        logging.debug(f"Boltz forward pass returned keys: {result_keys}")

        # Check for essential keys needed by the loss function
        expected_loss_keys = ['distogram', 'predicted_lddt', 'full_pae']
        missing_loss_keys = [k for k in expected_loss_keys if k not in boltz_result]
        if missing_loss_keys:
            # Raise error instead of warning, as these are critical for loss
            logging.error(f"Missing critical keys in boltz_result needed for loss: {missing_loss_keys}")
            raise KeyError(f"Model output missing required keys for Boltz loss: {missing_loss_keys}")

        # Check for distogram logits specifically
        if 'distogram' not in boltz_result or 'logits' not in boltz_result['distogram']:
             logging.error("Missing 'distogram[\'logits\']' in boltz_result.")
             raise KeyError("Model output missing 'distogram[\'logits\']' required for Boltz loss.")
        else:
             logging.debug(f"Distogram logits shape: {boltz_result['distogram']['logits'].shape}")

        logging.debug("Finished boltz_forward_pass.")
        return boltz_result

    def _update_opt_schedule(self, step, total_steps, base_opt):
        """Update optimizer schedule based on the current step and total steps."""
        # TODO: Implement more sophisticated scheduling (e.g., 3-stage)
        progress = step / max(1, total_steps - 1)
        updated_opt = {**base_opt} # Start with base options

        # Get start/end values from config
        boltz_cfg = self.config.boltz_config
        if not boltz_cfg:
            logging.warning("Boltz config not found, using default STE schedule values.")
            boltz_cfg = BoltzDesignConfig()

        temp_start = boltz_cfg.ste_temp_start
        temp_end = boltz_cfg.ste_temp_end
        soft_start = boltz_cfg.ste_soft_start
        soft_end = boltz_cfg.ste_soft_end
        hard_start = boltz_cfg.ste_hard_start
        hard_end = boltz_cfg.ste_hard_end

        updated_opt['temp'] = temp_start + progress * (temp_end - temp_start)
        updated_opt['soft'] = soft_start + progress * (soft_end - soft_start)
        updated_opt['hard'] = hard_start + progress * (hard_end - hard_start)
        updated_opt['alpha'] = boltz_cfg.ste_alpha # Use configured alpha

        return updated_opt

    def design(
        self,
        fold_input: folding_input.Input,
        feature_dict: features.BatchDict,
        target_indices: np.ndarray,
        binder_indices: np.ndarray,
        rng_key: jnp.ndarray,
        designer_params: Dict[str, jnp.ndarray],
        designer_inputs: Dict[str, Any],
        designer_opt: Dict[str, Any],
        optimizer: Optional[optax.GradientTransformation],
        optimizer_state: Optional[optax.OptState],
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """Design binder using a Boltzmann-inspired approach.

        Args:
            fold_input: Fold input containing target and binder.
            feature_dict: Feature dictionary.
            target_indices: Indices of the target residues.
            binder_indices: Indices of the binder residues.
            rng_key: Random key.
            designer_params: Dictionary containing designer parameters.
            designer_inputs: Dictionary containing designer inputs.
            designer_opt: Dictionary containing designer options.
            optimizer: Gradient transformation for optimization.
            optimizer_state: State of the optimizer.

        Returns:
            Tuple of (design_results, updated_feature_dict).
        """
        # Log initial config
        logging.info("=== Starting Boltz Design Trajectory ===")
        logging.info(f"Config: lr={self.config.learning_rate}, stages={self.config.boltz_config.stages}, best_metric={self.config.best_metric}")
        logging.info(f"Weights: {self.config.boltz_config.weights}")
        design_start_time = time.time()

        # Access config values directly
        helicity_bias = self.config.boltz_config.helicity_bias

        # Access config and compute random_init_scale
        bias = designer_inputs.get('bias')
        binder_length = len(binder_indices)
        random_init_scale = self.config.boltz_config.random_init_scale

        # Determine seed once for logging and reproducibility
        try:
            seed = int(jax.random.randint(rng_key, (), 0, 100000))
        except Exception:
            seed = int(np.sum(np.frombuffer(rng_key.tobytes(), dtype=np.uint32))) % 100000

        # Initialize sequence logits: use provided or random
        if 'seq_logits' in designer_params:
            binder_logits = designer_params['seq_logits']
            logging.info("Using provided seq_logits for Boltz design.")
        else:
            rng_key, subkey = jax.random.split(rng_key)
            binder_logits = random_init_scale * jax.random.normal(
                subkey, shape=(binder_length, ALPHABET_SIZE)
            )
            binder_logits = jnp.asarray(binder_logits, dtype=jnp.float32)

        # Log design parameters
        logging.info("Design parameters:")
        logging.info(f"- Protocol: Boltz")
        logging.info(f"- Binder Length: {binder_length}")
        logging.info(f"- Seed: {seed}")
        # logging.info(f"- Helicity Bias: {helicity_bias}") # Comment out until used

        # --- Setup for optimization ---
        # No further reinitialization of binder_logits needed
        rng_key, _ = jax.random.split(rng_key)

        # --- Apo/Holo Setup for Ligands --- >
        #logging.debug(f"Feature dict keys at start of design: {list(feature_dict.keys())}")
        feature_dict_template = copy.deepcopy(feature_dict) # Start with a mutable copy

        # Log all top-level keys of the copied template for debugging detection
        #logging.debug(f"DEBUG: Keys available in feature_dict_template: {list(feature_dict_template.keys())}")

        # Detect small-molecule binder scenario
        has_ligand = False
        if 'is_ligand' in feature_dict_template:
            try:
                # Convert to JAX array and check if any element is True/non-zero
                is_ligand_array = jnp.asarray(feature_dict_template['is_ligand'])
                if jnp.any(is_ligand_array):
                    has_ligand = True
                    logging.debug("Ligand detected based on non-zero values in 'is_ligand' array.")
                else:
                    logging.debug("'is_ligand' key found, but contains only zeros. No ligand detected.")
            except Exception as e:
                logging.warning(f"Could not process 'is_ligand' array for ligand detection: {e}")
        else:
            logging.debug("'is_ligand' key not found in features. No ligand detected.")

        apo_template_jax = None
        # THIS IS ONLY USED NOW MOMENTARILY FOR DEBUGGING
        has_ligand = False # THIS IS ONLY USED NOW MOMENTARILY FOR DEBUGGING
        # THIS IS ONLY USED NOW MOMENTARILY FOR DEBUGGING
        if has_ligand:
            logging.info("Small-molecule target detected: preparing apo -> holo two-phase schedule")
            # Apo phase: strip out ligand–binder coordinates/features
            apo_template = copy.deepcopy(feature_dict_template)
            keys_removed = []
            # Define keys to remove (data-related, keep boolean flags like is_ligand)
            keys_to_remove_for_apo = [
                'small_molecule_metadata', 'sm_metadata',
                'small_molecule_atom_positions', 'sm_atom_positions', 'ligand_atom_positions',
                'small_molecule_mask', 'sm_mask', 'ligand_mask',
                'ligand_features', 'sm_features',
                'residue_smiles',
                'polymer_ligand_bonds', # Bonds involving ligands
                'ligand_ligand_bonds'
                # Keep 'is_ligand' key itself, model might need it
            ]
            for k in keys_to_remove_for_apo:
                # Check against the specific removal list
                if k in apo_template:
                    del apo_template[k]
                    keys_removed.append(k)
            logging.info(f"Removed keys for apo template: {keys_removed}")
            apo_template_jax = freeze_containers_for_jax(apo_template)

        else:
            logging.info("No small-molecule target detected based on feature keys/token_types. Running standard single-phase schedule.")

        # Holo phase (or standard template if no ligand): full feature dict
        holo_template_jax = freeze_containers_for_jax(feature_dict_template)
        # <--- End Apo/Holo Setup ---

        # Define Stages based on config
        boltz_stages = self.config.boltz_config.stages
        stages_config = [
            {"name": "Warm-up", "iters": boltz_stages[0], "hard": False, "temp_range": (1.0, 1.0)},
            {"name": "Mixed-logit", "iters": boltz_stages[1], "hard": False, "temp_range": (1.0, 1.0)},
            {"name": "Anneal", "iters": boltz_stages[2], "hard": False, "temp_range": (1.0, 0.01)}, # Uses quadratic T schedule
            {"name": "STE", "iters": boltz_stages[3], "hard": True, "temp_range": (0.01, 0.01)} # Uses STE
            # PSSM stage removed
        ]
        # Only keep the number of stages defined by the length of the stages parameter
        num_defined_stages = len(boltz_stages)
        stages = stages_config[:num_defined_stages]

        logging.info(f"Using {num_defined_stages} stages for optimization:")
        for i, stage in enumerate(stages):
            logging.info(f"- Stage {i}: {stage['name']} ({stage['iters']} iters, Temp: {stage['temp_range'][0]:.2f}->{stage['temp_range'][1]:.2f}, Hard: {stage['hard']})" )

        # --- Prepare for optimization loop --- 
        # Convert indices for static compatibility if needed (JAX usually handles arrays fine)
        target_indices_jnp = jnp.array(target_indices)
        binder_indices_jnp = jnp.array(binder_indices)

        # Freeze feature dict template for JIT compatibility
        feature_dict_template_jax = freeze_containers_for_jax(feature_dict_template)
        model_runner_obj = self.model_runner # Passed by reference

        # Arguments to be passed to loss_fn_for_grad_boltz
        # Initialize custom args for loss function (used in holo phase)
        loss_fn_custom_args = {}

        # Define loss function for gradient calculation (inside design method)
        def loss_fn_for_grad_boltz(
            current_binder_logits,  # Differentiable
            stage_idx_static,         # Static
            temp_static,              # Static
            progress_static,          # Static (for stage 2 schedule)
            iter_key,                 # Varies
            feature_dict_tmpl,        # Static
            target_idxs,              # Static
            binder_idxs,              # Static
            weights_dict_static,      # Static
            model_runner_ref,       # Static
            current_opt: Dict[str, Any], # Current STE opt params
            current_bias: jnp.ndarray, # Current bias
            inter_k_arg: Optional[int] = None,
            inter_l_arg: Optional[int] = None,
        ):
            # 1. Get sequence representations using STE
            # Note: `current_opt` contains the scheduled temp, soft, hard, alpha
            seq_repr_dict = soft_seq_af3(
                current_binder_logits,
                current_bias,
                current_opt,
                iter_key
            )
 
            # 2. Update feature dictionary using the STE representations
            updated_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_tmpl,
                binder_idxs,
                seq_repr_dict # Pass the full dictionary
            )

            # Run the specialized forward pass
            # Use the correct template (apo or holo)
            boltz_result = self._boltz_forward_pass(updated_feature_dict, iter_key)

            # Calculate loss using the selected sequence representation
            total_loss, loss_breakdown = calculate_boltzdesign_binder_loss(
                boltz_result,
                updated_feature_dict,
                target_idxs,
                binder_idxs,
                weights_config=weights_dict_static,
                compute_confidence_loss=True, # Default unless overridden by mode
                inter_k=inter_k_arg,
                inter_l=inter_l_arg
            )

            # Add sequence entropy loss (always calculated based on weights_dict)
            seq_entropy_weight = weights_dict_static.seq_entropy
            if seq_entropy_weight > 0:
                # Use pssm from seq_repr_dict for entropy calculation
                pssm = seq_repr_dict['pssm']
                probs_clamped = jnp.clip(pssm, 1e-8, 1.0 - 1e-8)
                entropy_per_res = -jnp.sum(probs_clamped * jnp.log(probs_clamped), axis=-1)
                seq_entropy_loss = jnp.mean(entropy_per_res)
                total_loss += seq_entropy_weight * seq_entropy_loss
                loss_breakdown["seq_entropy"] = seq_entropy_loss

            return total_loss, loss_breakdown

        # Create gradient function (without JIT)
        grad_fn = jax.value_and_grad(loss_fn_for_grad_boltz, has_aux=True)

        # --- Optimization Loop --- 
        # Initialize optimizer and state for binder_logits array to avoid tree mismatch
        current_optimizer = optimizer if optimizer is not None else optax.adam(learning_rate=self.config.learning_rate)
        current_opt_state = current_optimizer.init(binder_logits)

        # --- Apo Warmup Phase (if ligand detected) ---
        logging.debug(f"DEBUG: Checking conditions for apo loop: has_ligand={has_ligand}, apo_template_jax is None: {apo_template_jax is None}") # <<< Explicit check before IF
        # For debugging purposes, set has_ligand to False even if ligand is detected! 
        has_ligand = False
        if has_ligand and apo_template_jax is not None:
            num_apo_steps = 1 # Reduced for debugging
            logging.info(f"---> Starting {num_apo_steps} apo-phase warmup iterations (no ligand) <---") # <<< Start of loop block
            # Temporarily override inter-contact weight to zero
            apo_weights = self.config.boltz_config.weights.copy()
            apo_weights['contact_inter'] = 0.0

            # Use grad_fn with apo template
            for apo_step in range(num_apo_steps):
                apo_iter_start_time = time.time()
                logging.debug(f"  DEBUG: Entering Apo iter {apo_step+1}/{num_apo_steps}")
                rng_key, iter_key = jax.random.split(rng_key)
                try:
                    (loss_val, loss_bd), grads = grad_fn(
                        binder_logits,
                        stage_idx_static=2, # Use softmax representation (Anneal stage logic)
                        temp_static=1.0,    # Fixed temp
                        progress_static=0.0, # Fixed progress
                        iter_key=iter_key,
                        # Static args
                        feature_dict_tmpl=apo_template_jax, # <-- Use APO template
                        target_idxs=target_indices_jnp,
                        binder_idxs=binder_indices_jnp,
                        weights_dict_static=apo_weights, # <-- Use APO weights
                        model_runner_ref=model_runner_obj,
                        current_opt=self._update_opt_schedule(0, 1, designer_opt),
                        current_bias=bias
                    )
                    # Log gradient norm to track STE during apo-phase
                    grad_norm = jnp.linalg.norm(grads)
                    logging.debug(f"  DEBUG: Apo iter {apo_step+1} gradient norm={safe_jax_to_float(grad_norm)}")
                    updates, current_opt_state = current_optimizer.update(grads, current_opt_state)
                    binder_logits = optax.apply_updates(binder_logits, updates)
                    # Log progress
                    logging.debug(f"  DEBUG: Apo iter {apo_step+1} loss calculated: {safe_jax_to_float(loss_val):.4f}") # <<< After calculation
                    if apo_step % 10 == (10 - 1):
                        apo_iter_time = time.time() - apo_iter_start_time
                        logging.info(f"  ---> Apo iter {apo_step+1}/{num_apo_steps} loss={safe_jax_to_float(loss_val):.3f} time={apo_iter_time:.2f}s <---") # <<< Periodic INFO log
                except Exception as e:
                    logging.error(f"Error during Apo iteration {apo_step}: {e}", exc_info=True)
                    break # Exit apo loop on error

            # --- Holo Phase Setup (if ligand detected) ---
            logging.info("---> Switching to holo phase: optimizing ligand contacts (k=1, l=1) <---") # <<< End of apo block / start of holo setup
            # Restore original weights (already in design_weights_static)
            # Set custom args for the loss function to trigger k=1,l=1 behavior
            loss_fn_custom_args = {'inter_k': 1, 'inter_l': 1}
            current_template_jax = holo_template_jax # Use holo template for main stages
        else:
            # If no ligand, just use the standard template
            current_template_jax = holo_template_jax # Holo template IS the standard template
            loss_fn_custom_args = {}
        # <--- End Apo/Holo Setup ---

        # Use the configured metric for tracking best
        best_metric_value = float('inf')
        metric_higher_is_better = self.config.best_metric in ["plddt_score"] # Add other metrics if needed
        best_logits = None
        best_metrics = None
        best_feature_dict = None # Store the feature dict corresponding to best_logits
        trajectory = {  # Initialize trajectory
            "step": [],
            "loss": [],
            "plddt": [],
            "pae_loss": [],
            "contact_inter": [],
            "contact_intra": [],
            "time": [],
            "log_history": [],
        }

        current_iter = 0
        for stage_idx, stage in enumerate(stages):
            stage_name = f"Stage {stage_idx}"
            stage_iters = stage['iters']
            start_temp, end_temp = stage['temp_range']
            hardness = stage['hard']
            # plddt_threshold = stage.get("plddt_threshold", 0.0)

            logging.info(f"Stage {stage_idx}: {stage_name} ({stage_iters} iters, Temp: {start_temp:.2f}->{end_temp:.2f}, Hard: {hardness})")

            for i in range(stage_iters):
                iter_start_time = time.time()
                progress = i / max(1, stage_iters - 1)
                temp = start_temp + progress * (end_temp - start_temp)
                temp = jnp.maximum(temp, 1e-4) # Ensure temp doesn't go too low

                rng_key, iter_key = jax.random.split(rng_key)

                try:
                    # Calculate current STE opt for this step using the schedule
                    total_iters_in_stage = stage_iters
                    current_opt = self._update_opt_schedule(i, total_iters_in_stage, designer_opt)

                    # Run gradient step
                    (loss_val, loss_breakdown), grads = grad_fn(
                        binder_logits,
                        stage_idx, # Static
                        temp,      # Static
                        progress,  # Static
                        iter_key,  # Varies
                        # Static args
                        current_template_jax,
                        target_indices_jnp,
                        binder_indices_jnp,
                        self.config.boltz_config.weights,
                        model_runner_obj,
                        current_opt,
                        bias,
                        inter_k_arg=loss_fn_custom_args.get('inter_k'),
                        inter_l_arg=loss_fn_custom_args.get('inter_l')
                    )

                    # Log gradient norm to track STE gradient flow
                    grad_norm = jnp.linalg.norm(grads)
                    logging.debug(f"Iter {current_iter}: Gradient norm={safe_jax_to_float(grad_norm)}")

                    if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                        logging.warning(f"Iter {current_iter}: Invalid loss ({loss_val}), skipping update.")
                        continue

                    # Update logits
                    # Let debug log a summary of the update
                    logging.debug(f"Iter {current_iter}: Updating logits with grads shape={grads.shape}")
                    updates, current_opt_state = current_optimizer.update(grads, current_opt_state)
                    binder_logits = optax.apply_updates(binder_logits, updates)
                    # Plot binder_logits probabilities at this iteration
                    _plot_logits(binder_logits, current_iter)
                    logging.debug(f"Iter {current_iter}: Logits updated, new shape={binder_logits.shape}")

                    # Log metrics (extract from loss_breakdown)
                    plddt_loss_neg_mean = loss_breakdown.get("plddt_neg_mean", 0.0)
                    plddt_score = safe_jax_to_float(-plddt_loss_neg_mean) # Actual pLDDT score (0-100)
                    pae_loss = loss_breakdown.get("pae_inter_mean", float('nan')) # Key used in boltzdesign loss
                    contact_inter = loss_breakdown.get("contact_inter", float('nan'))
                    contact_intra = loss_breakdown.get("contact_intra", float('nan'))
                    iter_time = time.time() - iter_start_time

                    # Adjust log format for pLDDT (0-100 scale)
                    log_line = (f"Iter {current_iter} [{stage_idx}] Loss={safe_jax_to_float(loss_val):.3f} "
                                f"pLDDT={plddt_score:.1f} PAE_loss={safe_jax_to_float(pae_loss):.3f} "
                                f"Cont_inter={safe_jax_to_float(contact_inter):.3f} Cont_intra={safe_jax_to_float(contact_intra):.3f} "
                                f"Temp={temp:.3f} Time={iter_time:.2f}s")
                    if current_iter % 10 == 0:
                         logging.info(log_line)
                    else:
                         logging.debug(log_line)

                    # Store trajectory
                    trajectory["step"].append(current_iter)
                    trajectory["loss"].append(safe_jax_to_float(loss_val))
                    trajectory["plddt"].append(plddt_score)
                    trajectory["pae_loss"].append(safe_jax_to_float(pae_loss))
                    trajectory["contact_inter"].append(safe_jax_to_float(contact_inter))
                    trajectory["contact_intra"].append(safe_jax_to_float(contact_intra))
                    trajectory["time"].append(iter_time)
                    trajectory["log_history"].append({
                        "loss": safe_jax_to_float(loss_val),
                        "plddt_score": plddt_score,
                        "pae_loss": safe_jax_to_float(pae_loss),
                        "contact_inter": safe_jax_to_float(contact_inter),
                        "contact_intra": safe_jax_to_float(contact_intra),
                        "ste_temp": current_opt['temp'],
                        "ste_soft": current_opt['soft'],
                        "ste_hard": current_opt['hard'],
                        "ste_alpha": current_opt['alpha']
                    })

                    # Extract the latest log entry for metric comparison
                    log_data = trajectory["log_history"][-1]
                    logging.debug(f"Iter {current_iter}: log_data={log_data}")
                    
                    # Check if this is the new best
                    current_metric_value = log_data.get(self.config.best_metric, float('inf'))

                    # Convert to minimization problem
                    comparison_value = -current_metric_value if metric_higher_is_better else current_metric_value

                    # Check if this is the new best
                    if comparison_value < best_metric_value:
                        best_metric_value = comparison_value
                        best_logits = binder_logits.copy()
                        best_metrics = log_data # Store the entire log dict as best metrics
                        # Regenerate feature dict for the best logits
                        logging.debug(f"Iter {current_iter}: New best found. Regenerating feature dict.")
                        # Recompute STE representations for the best logits
                        best_seq_repr = soft_seq_af3(
                            best_logits,
                            bias,
                            current_opt,
                            iter_key
                        )
                        best_feature_dict = binder_utils.update_features_from_logits(
                            current_template_jax,
                            binder_indices_jnp,
                            best_seq_repr
                        )
                        logging.info(f"Iter {current_iter}: New best {self.config.best_metric}={current_metric_value:.4f}")

                    current_iter += 1

                    # Memory clearing
                    if self.clear_memory_interval > 0 and current_iter % self.clear_memory_interval == 0:
                         logging.info(f"Iter {current_iter}: Executing scheduled memory clearing")
                         memory_utils.clear_memory(include_gpu=True, include_cpu=True)

                except Exception as e:
                    logging.error(f"Error in Boltz iteration {current_iter}: {e}", exc_info=True)
                    # Decide whether to continue or break
                    break # Safer to break the inner loop on error

            # --- End of stage --- 
            # Could add stage-end checks (like pLDDT threshold) here if needed

        # --- End of all stages --- 

        # Ensure best logits and feature dict are available
        if best_logits is None or best_metrics is None:
             logging.warning(f"Design finished without improving {self.config.best_metric}. Using final state.")
             best_logits = binder_logits
             # Try to get metrics from the last trajectory point
             best_metrics = trajectory["log_history"][-1] if trajectory["log_history"] else {"loss": float('nan')}
             best_metric_value = best_metrics.get(self.config.best_metric, float('nan'))
             if metric_higher_is_better: best_metric_value *= -1

        if best_feature_dict is None:
             logging.warning("Regenerating best_feature_dict from best_logits at the end.")
             # Recompute STE representations for fallback best_logits
             total_steps = sum(self.config.boltz_config.stages)
             final_opt = self._update_opt_schedule(total_steps - 1, total_steps, designer_opt)
             fallback_seq_repr = soft_seq_af3(
                 best_logits,
                 bias,
                 final_opt,
                 rng_key
             )
             best_feature_dict = binder_utils.update_features_from_logits(
                 current_template_jax,
                 binder_indices_jnp,
                 fallback_seq_repr
             )

        # Final sequence details
        final_probs = jax.nn.softmax(best_logits, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)
        aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        final_sequence = ''.join([aa_letters[int(i)] for i in final_aa_indices])

        # Calculate the final opt settings for the STE schedule
        total_steps = sum(self.config.boltz_config.stages)
        final_opt = self._update_opt_schedule(total_steps - 1, total_steps, designer_opt)

        # After optimization, prepare final feature dict
        final_seq_repr = soft_seq_af3(
            binder_logits,
            bias,
            final_opt, # Use the final scheduled opt settings
            rng_key
        )
        best_feature_dict = binder_utils.update_features_from_logits(
            freeze_containers_for_jax(feature_dict_template),
            jnp.array(binder_indices),
            final_seq_repr
        )

        design_results = {
            "protocol": "binder_boltz",
            "target_indices": target_indices,
            "binder_indices": binder_indices,
            "best_metric": self.config.best_metric,
            "best_metric_value": best_metric_value if not metric_higher_is_better else -best_metric_value,
            "best_metrics": best_metrics,
            "trajectory": trajectory,
            "design_time": time.time() - design_start_time,
            "final_seq_logits": best_logits,
            "final_aa_indices": [int(aa) for aa in final_aa_indices],
            "final_sequence": final_sequence,
            "best_feature_dict": best_feature_dict,
            # Return updated params and optimizer state for BinderDesigner
            "params": {'seq_logits': binder_logits},
            "optimizer_state": current_opt_state,
            "success": best_metrics.get("plddt_score", 0.0) > 70.0
        }

        logging.info(f"Boltz design completed in {design_results['design_time']:.2f}s")
        # Use the correct key for logging the final score
        logging.info(f"Best {self.config.best_metric}: {design_results['best_metric_value']:.4f}")
        logging.info(f"Final sequence: {final_sequence}")

        return design_results, best_feature_dict

    def run_final_prediction(
        self,
        fold_input: folding_input.Input,
        design_results: Dict[str, Any],
        rng_key: jnp.ndarray,
    ) -> Tuple[model.ModelResult, features.BatchDict, folding_input.Input]:
        """Run a final prediction with the designed sequence (Simplified)."""
        # This implementation can be identical to the one in GradientProtocol
        # TODO(felipe): Consolidate this method into the base class?
        # Copying the implementation here for completeness
        logging.info("Running final prediction with designed sequence (Simplified Method)...")

        required_keys = ["final_aa_indices", "best_feature_dict", "binder_indices", "final_seq_logits"]
        if not all(key in design_results for key in required_keys):
            logging.error(f"Missing required keys in design_results for final prediction: {[k for k in required_keys if k not in design_results]}")
            return {"error": "Missing design results"}, {}, fold_input

        try:
            final_aa_indices = design_results["final_aa_indices"]
            aa_types = 'ACDEFGHIKLMNPQRSTVWY'
            designed_sequence = ''.join([aa_types[int(idx)] for idx in final_aa_indices])
            logging.info(f"Designed sequence ({len(designed_sequence)} aa): {designed_sequence[:50]}...")

            new_chains = []
            # Access binder chains from config
            binder_chains = self.config.binder_chains
            if not binder_chains:
                logging.error("Binder chains not specified in config.")
                return {"error": "Missing binder chains"}, {}, fold_input

            binder_seq_by_chain = {}
            binder_start_idx = 0
            chain_lengths = {chain.id: len(chain) for chain in fold_input.chains}

            for binder_chain_id in binder_chains:
                if binder_chain_id in chain_lengths:
                    chain_length = chain_lengths[binder_chain_id]
                    if binder_start_idx + chain_length > len(designed_sequence):
                         logging.error(f"Sequence length mismatch for chain {binder_chain_id}")
                         return {"error": "Sequence length mismatch"}, {}, fold_input
                    binder_seq_by_chain[binder_chain_id] = designed_sequence[binder_start_idx:binder_start_idx + chain_length]
                    binder_start_idx += chain_length
                else:
                     logging.warning(f"Binder chain {binder_chain_id} not found.")

            for chain in fold_input.chains:
                if isinstance(chain, folding_input.ProteinChain) and chain.id in binder_seq_by_chain:
                    new_chain = folding_input.ProteinChain(
                        id=chain.id, sequence=binder_seq_by_chain[chain.id], ptms=chain.ptms,
                        unpaired_msa=None, paired_msa=None, templates=[]
                    )
                    new_chains.append(new_chain)
                else:
                    new_chains.append(chain)

            new_fold_input = folding_input.Input(
                name=fold_input.name + "_designed", chains=new_chains,
                rng_seeds=[fold_input.rng_seeds[0]] if fold_input.rng_seeds else [0],
                bonded_atom_pairs=fold_input.bonded_atom_pairs, user_ccd=fold_input.user_ccd
            )
            logging.info("Created new input (Simplified Method)")

            final_feature_dict = design_results["best_feature_dict"]
            if not final_feature_dict:
                 logging.error("Missing best feature dict.")
                 return {"error": "Missing best feature dict"}, {}, new_fold_input

            final_result = self.model_runner.run_inference(final_feature_dict, rng_key)
            logging.info("Final prediction complete (Simplified Method)")

            if isinstance(final_result, dict):
                if "metadata" not in final_result: final_result["metadata"] = {}
                final_result["metadata"]["designed_sequence"] = designed_sequence
            else:
                 final_result = {"output": final_result, "metadata": {"designed_sequence": designed_sequence}}

            return final_result, final_feature_dict, new_fold_input

        except KeyError as e:
            logging.error(f"KeyError: {e}", exc_info=True)
            return {"error": f"KeyError: {e}"}, {}, fold_input
        except Exception as e:
            logging.error(f"Error in final prediction: {e}", exc_info=True)
            return {"error": str(e)}, {}, fold_input 
"""Boltzmann-inspired binder design protocol."""

import time
import logging
import copy
from typing import Dict, Any, Tuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from alphafold3.common import folding_input
from alphafold3.common import memory_utils
from alphafold3.constants import chemical_components
from alphafold3.model import model
from alphafold3.model import features
from alphafold3.design import binder_utils
from alphafold3.design.losses import calculate_boltzdesign_binder_loss
from alphafold3.design.utils import freeze_containers_for_jax, safe_jax_to_float, safe_process_losses
from .base import BinderProtocol

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
        logging.debug(f"aatype shape: {feature_dict['aatype'].shape}, seq_mask shape: {feature_dict['seq_mask'].shape}")
        if 'msa' in feature_dict:
             logging.debug(f"msa shape: {feature_dict['msa'].shape}, msa_mask shape: {feature_dict.get('msa_mask', 'N/A')}")

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

    def design(
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
        logging.info("=== Starting Boltz Design Trajectory ===")
        design_start_time = time.time()

        # Log design parameters
        binder_length = len(binder_indices)
        try:
             seed = int(jax.random.randint(rng_key, (), 0, 100000))
        except Exception:
             seed = int(np.sum(np.frombuffer(rng_key.tobytes(), dtype=np.uint32))) % 100000
        helicity_bias = self.design_params.get("helicity_bias", 0.0) # Note: helicity loss not yet implemented

        logging.info(f"Design parameters:")
        logging.info(f"- Protocol: Boltz")
        logging.info(f"- Binder Length: {binder_length}")
        logging.info(f"- Seed: {seed}")
        # logging.info(f"- Helicity Bias: {helicity_bias}") # Comment out until used

        # --- Setup for optimization ---
        rng_key, subkey = jax.random.split(rng_key)

        # Initialize logits
        random_init_scale = self.design_params.get("random_init_scale", 0.01)
        num_residue_types = 20  # Standard amino acids
        binder_logits_shape = (len(binder_indices), num_residue_types)
        binder_logits = random_init_scale * jax.random.normal(
            subkey, shape=binder_logits_shape
        )
        binder_logits = jnp.asarray(binder_logits, dtype=jnp.float32)

        # Make a deep copy of the feature dictionary to avoid modifying the original
        # Use JAX compatible copy if possible, fallback to deepcopy
        try:
            feature_dict_template = jax.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, feature_dict)
        except Exception:
            logging.warning("Falling back to deepcopy for feature_dict_template")
            feature_dict_template = copy.deepcopy(feature_dict)

        # Extract weights for static compilation (ensure it's a standard dict)
        design_weights_static = dict(self.design_params.get("weights", {}))
        if not design_weights_static:
            logging.warning("No weights found in design_params['weights']. Using defaults.")
            # Provide default weights if missing
            design_weights_static = {
                'contact_intra': 1.0,
                'contact_inter': 1.0,
                'confidence': 0.5,
                'seq_entropy': 0.01 # Default needed by loss function
            }

        # --- Define Stages --- (Simplified logic from original implementation)
        # Defaulting to 4 core stages matching paper structure
        original_stages_len = len(self.design_params.get("stages", []))
        default_iters = [50, 50, 45, 5] # Default iters for 4 stages
        stages_config = [
            {"name": "Warm-up", "iters": self.design_params.get("stages", default_iters)[0] if original_stages_len>0 else 50, "hard": False, "temp_range": (1.0, 1.0)},
            {"name": "Mixed-logit", "iters": self.design_params.get("stages", default_iters)[1] if original_stages_len>1 else 50, "hard": False, "temp_range": (1.0, 1.0)},
            {"name": "Anneal", "iters": self.design_params.get("stages", default_iters)[2] if original_stages_len>2 else 45, "hard": False, "temp_range": (1.0, 0.01)}, # Uses quadratic T schedule
            {"name": "STE", "iters": self.design_params.get("stages", default_iters)[3] if original_stages_len>3 else 5, "hard": True, "temp_range": (0.01, 0.01)} # Uses STE
            # PSSM stage removed
        ]
        # Only keep the number of stages defined by the length of the stages parameter
        num_defined_stages = original_stages_len if original_stages_len > 0 else len(default_iters)
        stages = stages_config[:num_defined_stages]

        logging.info(f"Using {len(stages)} stages for optimization:")
        for i, stage in enumerate(stages):
            logging.info(f"- Stage {i}: {stage.get('name', 'Unknown')} ({stage.get('iters', 0)} iters)")

        # --- Prepare for optimization loop --- 
        # Convert indices for static compatibility if needed (JAX usually handles arrays fine)
        target_indices_jnp = jnp.array(target_indices)
        binder_indices_jnp = jnp.array(binder_indices)

        # Freeze feature dict template for JIT compatibility
        feature_dict_template_jax = freeze_containers_for_jax(feature_dict_template)
        model_runner_obj = self.model_runner # Passed by reference

        # Define loss function for gradient calculation (inside design method)
        def loss_fn_for_grad_boltz(
            current_binder_logits,  # Differentiable
            stage_idx_static,       # Static
            temp_static,            # Static
            progress_static,        # Static (for stage 2 schedule)
            iter_key,               # Varies
            # --- Potentially static arguments --- (Assumed static for now)
            feature_dict_tmpl,      # Static
            target_idxs,            # Static
            binder_idxs,            # Static
            design_weights,         # Static
            model_runner_ref        # Static
        ):
            # Determine sequence representation based on stage
            # Assuming 4 core stages: 0:Warm-up, 1:Mixed, 2:Anneal, 3:STE
            # (Indices adjusted from user feedback to match 0-based)

            logits_over_temp = current_binder_logits / temp_static

            # Stage 0: Warm-up (T=1.0)
            def stage_0_rep(_):
                return jax.nn.softmax(current_binder_logits, axis=-1) # Temp is 1.0 anyway

            # Stage 1: Mixed-logit (T=1.0, lambda based on progress)
            def stage_1_rep(prog):
                lambda_mix = prog # progress goes 0 -> 1
                return (1.0 - lambda_mix) * current_binder_logits + lambda_mix * jax.nn.softmax(current_binder_logits, axis=-1)

            # Stage 2: Anneal (Softmax with quadratic temp)
            def stage_2_rep(_):
                # Temp (temp_static here) is calculated externally with quadratic schedule
                return jax.nn.softmax(logits_over_temp, axis=-1)

            # Stage 3: STE (Hard one-hot with gradient trick)
            def stage_3_rep(key):
                probs = jax.nn.softmax(logits_over_temp, axis=-1)
                one_hot = jax.nn.one_hot(jnp.argmax(probs, axis=-1), num_classes=probs.shape[-1])
                # Straight-Through Estimator
                ste = jax.lax.stop_gradient(one_hot - probs) + probs
                return ste

            # Select representation based on stage index
            # We need to handle the arguments differently for each stage function
            if stage_idx_static == 0:
                seq_representation = stage_0_rep(None)
                gumbel_key_used = False
            elif stage_idx_static == 1:
                seq_representation = stage_1_rep(progress_static)
                gumbel_key_used = False
            elif stage_idx_static == 2:
                seq_representation = stage_2_rep(None)
                gumbel_key_used = False
            elif stage_idx_static == 3:
                # STE doesn't technically need a key, but gumbel_softmax version did
                seq_representation = stage_3_rep(iter_key)
                gumbel_key_used = False # STE is deterministic given logits
            else:
                # Fallback or error for unexpected stage index
                logging.warning(f"Unexpected stage index {stage_idx_static} in loss fn, using softmax.")
                seq_representation = jax.nn.softmax(current_binder_logits, axis=-1)
                gumbel_key_used = False

            # Update feature dictionary
            updated_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_tmpl, binder_idxs, seq_representation
            )

            # Run the specialized forward pass
            boltz_result = self._boltz_forward_pass(updated_feature_dict, iter_key) # Pass original key

            # Calculate loss using the selected sequence representation
            total_loss, loss_breakdown = calculate_boltzdesign_binder_loss(
                boltz_result,
                updated_feature_dict,
                target_idxs,
                binder_idxs,
                seq_representation, # Pass the representation used
                design_params={'weights': design_weights}
            )

            # Add sequence entropy loss based on soft probabilities in hard stage (STE stage)
            if stage_idx_static == 3:
                 # Use softmax of logits/temp for entropy calculation stability
                 probs_for_entropy = jax.nn.softmax(logits_over_temp, axis=-1)
                 probs_clamped = jnp.clip(probs_for_entropy, 1e-8, 1.0 - 1e-8)
                 # One-hot target derived from argmax of the same soft probabilities
                 one_hot_target = jax.nn.one_hot(jnp.argmax(probs_clamped, axis=-1), num_classes=probs_clamped.shape[-1])
                 one_hot_loss = -jnp.sum(one_hot_target * jnp.log(probs_clamped)) / len(binder_idxs)
                 seq_entropy_weight = design_weights.get("seq_entropy", 0.01)
                 total_loss += seq_entropy_weight * one_hot_loss
                 loss_breakdown["seq_one_hot_entropy"] = one_hot_loss * seq_entropy_weight

            return total_loss, loss_breakdown

        # Create gradient function (without JIT)
        grad_fn = jax.value_and_grad(loss_fn_for_grad_boltz, has_aux=True)

        # --- Optimization Loop --- 
        optimizer = optax.adam(learning_rate=self.design_params.get("learning_rate", 0.01)) # Use base LR for now
        opt_state = optimizer.init(binder_logits)

        best_loss = float('inf')
        best_logits = None
        best_metrics = None
        best_feature_dict = None # Store the feature dict corresponding to best_logits
        trajectory = {
            "step": [], "loss": [], "plddt": [], "pae_inter": [], "contact_inter": [],
             "contact_intra": [], "time": []
        }

        current_iter = 0
        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get("name", f"Stage {stage_idx}")
            stage_iters = stage.get("iters", 0)
            start_temp, end_temp = stage.get("temp_range", (1.0, 1.0))
            hardness = stage.get("hard", False)
            # plddt_threshold = stage.get("plddt_threshold", 0.0)

            # --- PSSM Stage Removed --- 
            # if stage_name == "PSSM-Greedy": # Handle PSSM stage separately
            #      logging.info(f"Stage {stage_idx}: Running PSSM Semigreedy Optimization ({stage_iters} iters)...")
            #      # ... (rest of PSSM logic removed) ...
            #      continue # Move to next stage after PSSM

            logging.info(f"Stage {stage_idx}: {stage_name} ({stage_iters} iters, Temp: {start_temp:.2f}->{end_temp:.2f}, Hard: {hardness})")

            for i in range(stage_iters):
                iter_start_time = time.time()
                progress = i / max(1, stage_iters - 1)
                temp = start_temp + progress * (end_temp - start_temp)
                temp = jnp.maximum(temp, 1e-4) # Ensure temp doesn't go too low

                rng_key, iter_key = jax.random.split(rng_key)

                try:
                    # Run gradient step
                    (loss_val, loss_breakdown), grads = grad_fn(
                        binder_logits,
                        stage_idx, # Static
                        temp,      # Static
                        progress,  # Static
                        iter_key,  # Varies
                        # Static args
                        feature_dict_template_jax,
                        target_indices_jnp,
                        binder_indices_jnp,
                        design_weights_static,
                        model_runner_obj
                    )

                    if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                        logging.warning(f"Iter {current_iter}: Invalid loss ({loss_val}), skipping update.")
                        continue

                    # Update logits
                    updates, opt_state = optimizer.update(grads, opt_state)
                    binder_logits = optax.apply_updates(binder_logits, updates)

                    # Log metrics (extract from loss_breakdown)
                    plddt = 1.0 - loss_breakdown.get("plddt", 1.0) # Convert loss back to score
                    pae_inter = loss_breakdown.get("pae_inter", float('nan'))
                    contact_inter = loss_breakdown.get("contact_inter", float('nan'))
                    contact_intra = loss_breakdown.get("contact_intra", float('nan'))
                    iter_time = time.time() - iter_start_time

                    log_line = (f"Iter {current_iter} [{stage_idx}] Loss={safe_jax_to_float(loss_val):.3f} "
                                f"pLDDT={plddt:.3f} PAE_inter={safe_jax_to_float(pae_inter):.3f} "
                                f"Cont_inter={safe_jax_to_float(contact_inter):.3f} Cont_intra={safe_jax_to_float(contact_intra):.3f} "
                                f"Temp={temp:.3f} Time={iter_time:.2f}s")
                    if current_iter % 10 == 0:
                         logging.info(log_line)
                    else:
                         logging.debug(log_line)

                    # Store trajectory
                    trajectory["step"].append(current_iter)
                    trajectory["loss"].append(safe_jax_to_float(loss_val))
                    trajectory["plddt"].append(plddt)
                    trajectory["pae_inter"].append(safe_jax_to_float(pae_inter))
                    trajectory["contact_inter"].append(safe_jax_to_float(contact_inter))
                    trajectory["contact_intra"].append(safe_jax_to_float(contact_intra))
                    trajectory["time"].append(iter_time)

                    # Check if this is the best loss so far
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_logits = binder_logits.copy()
                        best_metrics = {
                            "loss": safe_jax_to_float(loss_val),
                            "plddt": plddt,
                            "pae_inter": safe_jax_to_float(pae_inter),
                            "contact_inter": safe_jax_to_float(contact_inter),
                            "contact_intra": safe_jax_to_float(contact_intra)
                        }
                        # Regenerate feature dict for the best logits
                        logging.debug(f"Iter {current_iter}: New best found. Regenerating feature dict.")
                        best_feature_dict = binder_utils.update_features_from_logits(
                             feature_dict_template_jax, binder_indices_jnp, best_logits
                        )
                        logging.info(f"Iter {current_iter}: New best loss={best_metrics['loss']:.4f} pLDDT={best_metrics['plddt']:.3f}")

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
        if best_logits is None:
             logging.warning("Design finished without ever improving initial loss. Using final logits.")
             best_logits = binder_logits
             best_loss = trajectory["loss"][-1] if trajectory["loss"] else float('inf')
             best_metrics = {k: v[-1] for k, v in trajectory.items() if k != 'step' and v} if trajectory["loss"] else {}

        if best_feature_dict is None:
             logging.warning("Regenerating best_feature_dict from best_logits at the end.")
             best_feature_dict = binder_utils.update_features_from_logits(
                 feature_dict_template_jax, binder_indices_jnp, best_logits
             )

        # Final sequence details
        final_probs = jax.nn.softmax(best_logits, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)
        aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        final_sequence = ''.join([aa_letters[int(i)] for i in final_aa_indices])

        design_results = {
            "protocol": "binder_boltz",
            "target_indices": target_indices,
            "binder_indices": binder_indices,
            "best_loss": safe_jax_to_float(best_loss),
            "best_metrics": best_metrics,
            "trajectory": trajectory,
            "design_time": time.time() - design_start_time,
            "final_seq_logits": best_logits,
            "final_aa_indices": [int(aa) for aa in final_aa_indices],
            "final_sequence": final_sequence,
            "best_feature_dict": best_feature_dict,
            "success": best_metrics.get("plddt", 0.0) > 0.7 # Example success criterion
        }

        logging.info(f"Boltz design completed in {design_results['design_time']:.2f}s")
        logging.info(f"Best loss: {design_results['best_loss']:.4f}, Final pLDDT: {best_metrics.get('plddt', 0.0):.3f}")
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
            binder_chains = self.design_params.get("binder_chains", [])
            if not binder_chains:
                logging.error("Binder chains not specified.")
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
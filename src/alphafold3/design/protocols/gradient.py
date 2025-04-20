"""Gradient-based binder design protocol."""

import time
import logging
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
from alphafold3.design.losses import calculate_gradient_binder_loss
from alphafold3.design.utils import safe_jax_to_float, safe_process_losses
from .base import BinderProtocol

class GradientProtocol(BinderProtocol):
    """Implements the gradient-based binder design protocol."""

    def design(
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
            jax.random.PRNGKey(0), binder_seq_logits.shape # Use a fixed key for init
        )

        # Setup optimizer
        lr = self.design_params.get("lr", 0.1)
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(binder_seq_logits)

        # Filter out non-JAX compatible types from feature_dict for JIT
        keys_to_remove = [k for k, v in feature_dict.items() if isinstance(v, np.ndarray) and v.dtype == object]
        logging.debug(f"Filtering object-dtype keys from feature_dict for JIT: {keys_to_remove}")
        feature_dict_jax = {k: v for k, v in feature_dict.items() if k not in keys_to_remove}
        logging.debug(f"Feature_dict_jax keys: {list(feature_dict_jax.keys())}")

        # Setup logging and trajectory storage
        trajectory = {
            "loss": [],
            "losses": [],
            "step": [],
            "time": [],
            "sequences": [],
        }

        # Define loss function for gradient calculation (inside design method)
        def loss_fn_for_grad(curr_binder_logits, feature_dict_template, target_indices_static,
                            binder_indices_static, design_params_static, model_runner_obj, current_rng_key):
            # Convert static tuples back if needed (though JAX usually handles arrays)
            target_indices_arr = jnp.array(target_indices_static)
            binder_indices_arr = jnp.array(binder_indices_static)

            # Update features based on logits
            # Ensure feature_dict_template is treated as immutable if passed directly
            # Using .copy() might be safer if modifications happen inside update_features
            updated_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_template, binder_indices_arr, curr_binder_logits
            )

            # Run full forward pass
            result = model_runner_obj.run_inference(updated_feature_dict, current_rng_key)

            # Calculate loss
            total_loss, loss_breakdown = calculate_gradient_binder_loss(
                result, updated_feature_dict, target_indices_arr, binder_indices_arr,
                curr_binder_logits, design_params_static
            )

            # Return loss and breakdown for value_and_grad
            return total_loss, loss_breakdown

        # Create gradient function (without JIT decoration)
        grad_fn = jax.value_and_grad(loss_fn_for_grad, has_aux=True)

        # Standard optimization loop
        best_loss = float('inf')
        best_logits = None
        best_feature_dict = None
        steps = self.design_params.get("steps", 200)

        # Convert numpy arrays to JAX arrays for JIT compatibility if used inside grad_fn
        target_indices_jnp = jnp.array(target_indices)
        binder_indices_jnp = jnp.array(binder_indices)

        for step in range(steps):
            step_start_time = time.time()
            step_key, rng_key = jax.random.split(rng_key)

            # Compute gradients
            # Pass static args. Note: design_params dict might not be JAX-traceable directly
            # Consider passing only necessary weights or making it a static arg if JIT applied here
            (loss, losses), grads = grad_fn(
                binder_seq_logits,      # Differentiable
                feature_dict_jax,       # Static template
                target_indices_jnp,     # Static indices
                binder_indices_jnp,     # Static indices
                self.design_params,     # Static parameters (might need freezing/subsetting for JIT)
                self.model_runner,      # Static object (passed by reference)
                step_key                # Varies
            )

            # Update logits
            updates, opt_state = optimizer.update(grads, opt_state, binder_seq_logits)
            binder_seq_logits = optax.apply_updates(binder_seq_logits, updates)

            # Get loss value for logging
            step_loss_val = loss
            step_losses_val = losses

            # Log progress
            step_time = time.time() - step_start_time
            if step % 10 == 0 or step == steps - 1:
                probs = jax.nn.softmax(binder_seq_logits, axis=-1)
                aa_indices = jnp.argmax(probs, axis=-1)

                # Convert to amino acid sequence
                aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
                aa_indices_py = [int(idx) for idx in aa_indices]
                current_seq = ''.join([aa_letters[idx] for idx in aa_indices_py])

                logging.info(f"Step {step}/{steps}: loss={safe_jax_to_float(step_loss_val):.4f}, time={step_time:.2f}s")
                safe_losses = safe_process_losses(step_losses_val)
                logging.info(f"Losses: {safe_losses}")

                trajectory["sequences"].append(current_seq)

            # Store in trajectory (potentially large if storing JAX arrays)
            # Consider storing only scalar loss values
            trajectory["loss"].append(safe_jax_to_float(step_loss_val))
            trajectory["losses"].append(safe_process_losses(step_losses_val))
            trajectory["step"].append(step)
            trajectory["time"].append(step_time)

            # Track best loss
            if step_loss_val < best_loss:
                best_loss = step_loss_val # Keep as JAX array for comparison
                best_logits = binder_seq_logits

                # Regenerate best feature dict only when a new best is found
                logging.debug(f"Step {step}/{steps}: Regenerating feature dict for new best solution")
                best_feature_dict = binder_utils.update_features_from_logits(
                    feature_dict_jax, # Use the JAX-compatible version
                    binder_indices_jnp,
                    best_logits
                )
                logging.info(f"Step {step}/{steps}: New best loss {safe_jax_to_float(best_loss):.4f}")

            # Periodically clear memory if enabled
            if self.clear_memory_interval > 0 and step % self.clear_memory_interval == 0 and step > 0:
                logging.info(f"Step {step}/{steps}: Executing scheduled memory clearing")
                mem_usage_before = memory_utils.get_current_memory_usage()
                if mem_usage_before:
                    logging.info(f"Before memory clearing: {mem_usage_before}")

                memory_utils.clear_memory(include_gpu=True, include_cpu=True)

                mem_usage_after = memory_utils.get_current_memory_usage()
                if mem_usage_after:
                    logging.info(f"After memory clearing: {mem_usage_after}")

        # Final sequence generation from best logits
        if best_logits is None:
             logging.warning("No best logits found, design may have failed. Using last logits.")
             best_logits = binder_seq_logits # Fallback

        final_probs = jax.nn.softmax(best_logits, axis=-1)
        final_aa_indices = jnp.argmax(final_probs, axis=-1)

        # If best_feature_dict wasn't generated (e.g., loss never improved)
        if best_feature_dict is None:
             logging.warning("Best feature dict not generated, regenerating from final logits.")
             best_feature_dict = binder_utils.update_features_from_logits(
                 feature_dict_jax,
                 binder_indices_jnp,
                 best_logits
             )

        # Prepare results
        design_results = {
            "protocol": "binder_gradient",
            "target_indices": target_indices, # Store original numpy arrays
            "binder_indices": binder_indices,
            "best_loss": safe_jax_to_float(best_loss), # Store scalar float
            "trajectory": trajectory,
            "design_time": time.time() - design_start_time,
            "final_seq_logits": best_logits,
            "final_aa_indices": [int(aa) for aa in final_aa_indices], # Convert to Python list
            # Keep placeholders for compatibility if needed by final prediction
            "best_feature_dict": best_feature_dict, # The dict corresponding to best_logits
            "success": True # Assume success unless specific failure condition met
        }

        logging.info(f"Gradient-based binder design completed in {design_results['design_time']:.2f}s")
        logging.info(f"Best loss: {design_results['best_loss']:.4f}")

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

        # Check for essential keys in design_results
        required_keys = ["final_aa_indices", "best_feature_dict", "binder_indices", "final_seq_logits"]
        if not all(key in design_results for key in required_keys):
            logging.error(f"Missing required keys in design_results for final prediction: {[k for k in required_keys if k not in design_results]}")
            return {"error": "Missing design results"}, {}, fold_input

        try:
            # Get the final amino acid indices from the design results
            final_aa_indices = design_results["final_aa_indices"]

            # Convert to amino acid sequence
            aa_types = 'ACDEFGHIKLMNPQRSTVWY'
            designed_sequence = ''.join([aa_types[int(idx)] for idx in final_aa_indices])
            logging.info(f"Designed sequence ({len(designed_sequence)} aa): {designed_sequence[:50]}...")

            # Create a new fold_input with the designed sequence
            new_chains = []
            binder_chains = self.design_params.get("binder_chains", [])
            if not binder_chains:
                logging.error("Binder chains not specified in design_params. Cannot create new input.")
                return {"error": "Missing binder chains"}, {}, fold_input

            # Create a mapping of chain ID to the new designed sequence
            binder_seq_by_chain = {}
            binder_start_idx = 0
            chain_lengths = {chain.id: len(chain) for chain in fold_input.chains}

            for binder_chain_id in binder_chains:
                if binder_chain_id in chain_lengths:
                    chain_length = chain_lengths[binder_chain_id]
                    if binder_start_idx + chain_length > len(designed_sequence):
                         logging.error(f"Designed sequence length error for chain {binder_chain_id}")
                         return {"error": "Sequence length mismatch"}, {}, fold_input
                    binder_seq_by_chain[binder_chain_id] = designed_sequence[binder_start_idx:binder_start_idx + chain_length]
                    binder_start_idx += chain_length
                else:
                     logging.warning(f"Binder chain {binder_chain_id} not found in input.")

            for chain in fold_input.chains:
                if isinstance(chain, folding_input.ProteinChain) and chain.id in binder_seq_by_chain:
                    new_sequence = binder_seq_by_chain[chain.id]
                    new_chain = folding_input.ProteinChain(
                        id=chain.id, sequence=new_sequence, ptms=chain.ptms,
                        unpaired_msa=None, paired_msa=None, templates=[]
                    )
                    new_chains.append(new_chain)
                    logging.info(f"Replaced binder chain {chain.id} sequence.")
                else:
                    new_chains.append(chain)

            new_fold_input = folding_input.Input(
                name=fold_input.name + "_designed",
                chains=new_chains,
                rng_seeds=[fold_input.rng_seeds[0]] if fold_input.rng_seeds else [0],
                bonded_atom_pairs=fold_input.bonded_atom_pairs,
                user_ccd=fold_input.user_ccd
            )
            logging.info("Created new input with designed binder sequence (Simplified Method)")

            # Use the best feature dict corresponding to the best logits
            final_feature_dict = design_results["best_feature_dict"]
            if not final_feature_dict:
                 logging.error("Best feature dict is empty or missing.")
                 return {"error": "Missing best feature dict"}, {}, new_fold_input

            # Ensure the feature dict corresponds to the final sequence if best_feature_dict logic failed
            # It's generally better to rely on best_feature_dict being correct
            # Re-calculating here can be redundant if done right in the design loop
            # final_feature_dict = binder_utils.update_features_from_logits(
            #     final_feature_dict, # Start from the best one
            #     jnp.array(design_results["binder_indices"]),
            #     design_results["final_seq_logits"]
            # )

            # Run standard prediction
            logging.info("Running inference on designed sequence (Simplified Method)...")
            final_result = self.model_runner.run_inference(final_feature_dict, rng_key)
            logging.info("Final prediction complete (Simplified Method)")

            # Add designed sequence to the result metadata
            if isinstance(final_result, dict):
                if "metadata" not in final_result: final_result["metadata"] = {}
                final_result["metadata"]["designed_sequence"] = designed_sequence
            else:
                 logging.warning(f"Final model result is not a dict: {type(final_result)}.")
                 final_result = {"output": final_result, "metadata": {"designed_sequence": designed_sequence}}

            return final_result, final_feature_dict, new_fold_input

        except KeyError as e:
            logging.error(f"KeyError during final prediction: Missing key {e} in design_results", exc_info=True)
            return {"error": f"KeyError: {e}"}, {}, fold_input
        except Exception as e:
            logging.error(f"Error in final prediction: {e}", exc_info=True)
            return {"error": str(e)}, {}, fold_input 
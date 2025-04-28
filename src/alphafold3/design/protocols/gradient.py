"""Gradient-based binder design protocol."""

import time
import logging
from typing import Dict, Any, Tuple, Sequence, Optional

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
from alphafold3.design.utils import safe_jax_to_float, safe_process_losses, print_log_line
from .base import BinderProtocol
from alphafold3.design.sequence_utils import soft_seq_af3
from alphafold3.design.config import ALPHABET_SIZE, GradientDesignConfig

class GradientProtocol(BinderProtocol):
    """Implements the gradient-based binder design protocol."""

    def _update_opt_schedule(self, step: int, total_steps: int, base_opt: Dict[str, Any]) -> Dict[str, Any]:
        """Updates optimization parameters based on a schedule (linear ramp)."""
        # TODO: Implement more sophisticated scheduling (e.g., 3-stage)
        progress = step / max(1, total_steps - 1)
        updated_opt = {**base_opt} # Start with base options

        grad_cfg = self.config.gradient_config
        if not grad_cfg:
            # Should not happen if protocol is gradient, but handle gracefully
            logging.warning("Gradient config not found, using default STE schedule values.")
            grad_cfg = GradientDesignConfig() # Use default values

        temp_start = grad_cfg.ste_temp_start
        temp_end = grad_cfg.ste_temp_end
        soft_start = grad_cfg.ste_soft_start
        soft_end = grad_cfg.ste_soft_end
        hard_start = grad_cfg.ste_hard_start
        hard_end = grad_cfg.ste_hard_end

        updated_opt['temp'] = temp_start + progress * (temp_end - temp_start)
        updated_opt['soft'] = soft_start + progress * (soft_end - soft_start)
        updated_opt['hard'] = hard_start + progress * (hard_end - hard_start)
        # Alpha often remains constant
        updated_opt['alpha'] = base_opt.get('alpha', 1.0)

        # Ensure alpha from config is used if present
        updated_opt['alpha'] = grad_cfg.ste_alpha

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
        """Gradient-based binder design using full model and STE.

        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The initial feature dictionary.
            target_indices: Indices of target residues.
            binder_indices: Indices of binder residues.
            rng_key: JAX random key.
            designer_params: Dictionary containing trainable parameters (e.g., {'seq_logits': ...}).
            designer_inputs: Dictionary containing fixed inputs (e.g., {'bias': ...}).
            designer_opt: Dictionary containing optimization settings (temp, alpha, soft, hard).
            optimizer: Optax optimizer instance.
            optimizer_state: Current state of the optimizer.

        Returns:
            Tuple of (design_results, final_feature_dict, updated_params, updated_optimizer_state).
        """
        logging.info(f"Starting STE gradient-based binder design (lr={self.config.learning_rate}, steps={self.config.gradient_config.steps})...")
        design_start_time = time.time()

        if not designer_params or 'seq_logits' not in designer_params:
             raise ValueError("'seq_logits' must be initialized in designer_params.")
        if not designer_inputs or 'bias' not in designer_inputs:
             logging.warning("Bias not found in designer_inputs, defaulting to zeros.")
             binder_length = designer_params['seq_logits'].shape[0]
             designer_inputs['bias'] = jnp.zeros((binder_length, ALPHABET_SIZE), dtype=jnp.float32)

        current_params = designer_params
        current_inputs = designer_inputs
        current_opt = designer_opt
        current_optimizer = optimizer
        current_opt_state = optimizer_state

        if current_optimizer is None or current_opt_state is None:
            raise ValueError("Optimizer and optimizer state must be set and initialized before calling design.")

        keys_to_remove = [k for k, v in feature_dict.items() if isinstance(v, np.ndarray) and v.dtype == object]
        logging.debug(f"Filtering object-dtype keys from feature_dict for JIT: {keys_to_remove}")
        feature_dict_template_jax = {k: v for k, v in feature_dict.items() if k not in keys_to_remove}
        logging.debug(f"Feature_dict_template_jax keys: {list(feature_dict_template_jax.keys())}")

        trajectory = {
            "step": [], "loss": [], "losses": [], "time": [],
            "seq_logits": [], "hard_seq": [], "pseudo_seq_prob": [],
            "grad_norm": []
        }

        @jax.jit
        def compute_loss_and_grads(params, inputs, opt, feature_dict_tmpl, target_idxs_static,
                                 binder_idxs_static, weights_dict_static, model_runner_ref, current_rng_key):
            logits = params['seq_logits']
            bias = inputs['bias']
            iter_key, model_key = jax.random.split(current_rng_key)

            seq_repr_dict = soft_seq_af3(logits, bias, opt, iter_key)

            updated_feature_dict = binder_utils.update_features_from_logits(
                feature_dict_tmpl, binder_idxs_static, seq_repr_dict
            )

            result = model_runner_ref.run_inference(updated_feature_dict, model_key)

            total_loss, loss_breakdown = calculate_gradient_binder_loss(
                result,
                updated_feature_dict,
                target_idxs_static,
                binder_idxs_static,
                seq_repr_dict['logits'],
                weights_dict_static
            )

            aux_data = {
                "loss_breakdown": loss_breakdown,
                "seq_repr": seq_repr_dict,
            }
            return total_loss, aux_data

        grad_fn = jax.value_and_grad(compute_loss_and_grads, has_aux=True, argnums=0)

        best_loss = float('inf')
        best_metric_value = float('inf')
        best_metric_comparison_value = float('inf')
        metric_higher_is_better = self.config.best_metric in ["plddt_score"]

        best_logits = None
        best_feature_dict_final = None
        steps = self.config.gradient_config.steps

        target_indices_jnp = jnp.array(target_indices)
        binder_indices_jnp = jnp.array(binder_indices)

        for step in range(steps):
            step_start_time = time.time()
            step_key, rng_key = jax.random.split(rng_key)

            current_opt = self._update_opt_schedule(step, steps, current_opt)

            w = self.config.gradient_config.weights
            weights_dict_static = {
                "plddt": w.gradient_plddt,
                "pae_inter": w.gradient_pae_inter,
                "contact_inter": w.gradient_contact_inter,
                "contact_intra": w.gradient_contact_intra,
                "fape_target": w.gradient_fape_target,
                "seq_entropy": w.seq_entropy
            }

            try:
                 (loss_val, aux_data), grads = grad_fn(
                     current_params,
                     current_inputs,
                     current_opt,
                     feature_dict_template_jax,
                     target_indices_jnp,
                     binder_indices_jnp,
                     weights_dict_static,
                     self.model_runner,
                     step_key
                 )
            except Exception as e:
                 logging.error(f"Error during gradient calculation at step {step}: {e}", exc_info=True)
                 design_results = {"protocol": self.protocol, "success": False, "error": str(e)}
                 return design_results, feature_dict

            if 'seq_logits' not in grads:
                logging.error(f"Gradients calculation failed: 'seq_logits' not found in grads dict at step {step}. Keys: {grads.keys()}")
                design_results = {"protocol": self.protocol, "success": False, "error": "Gradient calculation failed."}
                return design_results, feature_dict

            updates, current_opt_state = current_optimizer.update(grads, current_opt_state, current_params)
            current_params = optax.apply_updates(current_params, updates)

            step_time = time.time() - step_start_time
            grad_norm = jnp.linalg.norm(jax.tree_util.tree_leaves(grads['seq_logits']))

            seq_repr = aux_data['seq_repr']
            hard_seq_indices = jnp.argmax(seq_repr['hard'], axis=-1)
            pseudo_seq_probs = seq_repr['pseudo']

            log_data = {
                "step": step,
                "loss": safe_jax_to_float(loss_val),
                "grad_norm": safe_jax_to_float(grad_norm),
                **safe_process_losses(aux_data['loss_breakdown'])
            }

            if step % self.config.verbosity == 0 or step == steps - 1:
                print_log_line(f"Step {step}/{steps}", log_data)

            trajectory["step"].append(step)
            trajectory["loss"].append(log_data["loss"])
            trajectory["losses"].append({k: v for k, v in log_data.items() if k not in ["step", "loss", "grad_norm"]})
            trajectory["time"].append(step_time)
            trajectory["seq_logits"].append(np.array(current_params['seq_logits']))
            trajectory["hard_seq"].append(np.array(hard_seq_indices))
            trajectory["pseudo_seq_prob"].append(np.array(pseudo_seq_probs))
            trajectory["grad_norm"].append(log_data["grad_norm"])

            current_metric_value = log_data.get(self.config.best_metric, log_data["loss"])
            comparison_value = -current_metric_value if metric_higher_is_better else current_metric_value

            if comparison_value < best_metric_comparison_value:
                best_metric_comparison_value = comparison_value
                best_metric_value = current_metric_value
                best_loss = log_data["loss"]
                best_logits = current_params['seq_logits']
                logging.info(f"Step {step}/{steps}: New best {self.config.best_metric}={best_metric_value:.4f} (Loss={best_loss:.4f})")

            if self.clear_memory_interval > 0 and step % self.clear_memory_interval == 0 and step > 0:
                logging.info(f"Step {step}/{steps}: Executing scheduled memory clearing")
                memory_utils.clear_memory(include_gpu=True, include_cpu=True)

        if best_logits is None:
            logging.warning("No best logits found, design may have failed. Using last logits.")
            best_logits = current_params['seq_logits']

        logging.info("Regenerating final feature dict from best logits...")
        final_seq_repr_dict = soft_seq_af3(
            best_logits,
            current_inputs['bias'],
            current_opt,
            rng_key
        )
        best_feature_dict_final = binder_utils.update_features_from_logits(
            feature_dict_template_jax,
            binder_indices_jnp,
            final_seq_repr_dict
        )

        best_logits_np = np.array(best_logits)
        final_aa_indices_np = np.array(jnp.argmax(jax.nn.softmax(best_logits, axis=-1), axis=-1))

        design_results = {
            "protocol": self.protocol,
            "steps": steps,
            "final_seq_logits": best_logits_np,
            "trajectory": trajectory,
            "best_metric": self.config.best_metric,
            "best_metric_value": best_metric_value,
            "best_loss": best_loss,
            "target_indices": target_indices,
            "binder_indices": binder_indices,
            "design_time": time.time() - design_start_time,
            "final_aa_indices": [int(aa) for aa in final_aa_indices_np],
            "best_feature_dict": best_feature_dict_final,
            "success": True,
            "params": current_params,
            "optimizer_state": current_opt_state,
        }

        logging.info(f"STE Gradient-based binder design completed in {design_results['design_time']:.2f}s")
        logging.info(f"Best {design_results['best_metric']}: {design_results['best_metric_value']:.4f} (Loss: {design_results['best_loss']:.4f})")

        return design_results, best_feature_dict_final

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
            binder_chains = self.config.binder_chains
            if not binder_chains:
                logging.error("Binder chains not specified in config. Cannot create new input.")
                return {"error": "Missing binder chains"}, {}, fold_input

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

            final_feature_dict = design_results["best_feature_dict"]
            if not final_feature_dict:
                 logging.error("Best feature dict is empty or missing.")
                 return {"error": "Missing best feature dict"}, {}, new_fold_input

            logging.info("Running inference on designed sequence (Simplified Method)...")
            final_result = self.model_runner.run_inference(final_feature_dict, rng_key)
            logging.info("Final prediction complete (Simplified Method)")

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
"""Binder design main entry point and factory class."""

import time
import functools
from typing import Dict, Any, Tuple, Sequence, Optional
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
from alphafold3.design.losses import (
    # Common utils (use sparingly)
    entropy_low_bins,
    safe_mean,
    
    # Specific loss components
    get_binder_plddt_loss,
    calculate_plddt_loss,
    calculate_plddt_confidence_weighted_loss,
    get_interface_pae_loss,
    calculate_pae_loss,
    calculate_pae_confidence_loss,
    calculate_max_interface_pae_loss,
    get_distogram_entropy_loss,
    get_binder_seq_entropy_loss,
    get_target_fape_loss,
    
    # Combined protocol losses
    calculate_gradient_binder_loss,
    calculate_boltzdesign_binder_loss, # Note: renamed from calculate_boltz_binder_loss
)
# Import protocols
from alphafold3.design.protocols import BinderProtocol, GradientProtocol, BoltzProtocol
from alphafold3.design.config import DesignConfig
from alphafold3.design import plotting

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
    """Factory class to select and run binder design protocols."""

    def __init__(
        self,
        model_runner: Any,
        ccd: chemical_components.Ccd,
        design_config: DesignConfig,
    ):
        """Initialize the binder designer factory with a DesignConfig.
        
        Args:
            model_runner: ModelRunner instance.
            ccd: Chemical component dictionary.
            design_config: Structured design configuration.
        """
        self.model_runner = model_runner
        self.ccd = ccd
        self.config = design_config
        # Use protocol_name from config
        self.protocol_name = self.config.protocol_name
        self.protocol_instance = self._create_protocol()

        logging.info(f"Initialized BinderDesigner with protocol: {self.protocol_name}")
        logging.debug(f"Design config: {self.config}")

    def _create_protocol(self) -> BinderProtocol:
        """Instantiates the correct protocol based on design_config."""
        if self.protocol_name == "binder_gradient":
            return GradientProtocol(self.model_runner, self.ccd, self.config)
        elif self.protocol_name == "binder_boltz":
            return BoltzProtocol(self.model_runner, self.ccd, self.config)
        else:
            raise ValueError(f"Unknown design protocol: {self.protocol_name}")
    
    def design_binder(
        self, 
        fold_input: folding_input.Input,
        feature_dict: features.BatchDict,
        rng_key: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """Design a binder protein using the selected protocol.
        
        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The feature dictionary for the model.
            rng_key: JAX random key.
            
        Returns:
            Tuple of (design_results, final_feature_dict).
        """
        # Use chains from config
        target_chains = self.config.target_chains
        binder_chains = self.config.binder_chains

        # Modify feature_dict for binder design (common setup)
        feature_dict, target_indices, binder_indices = binder_utils.setup_binder_features(
            feature_dict, target_chains, binder_chains, fold_input=fold_input
        )

        # Store indices for potential use
        self._target_indices = target_indices
        self._binder_indices = binder_indices
        
        # Convert necessary numpy arrays in feature_dict to JAX arrays for gradient updates
        # This might be better handled within the protocols if needed differently
        if 'aatype' in feature_dict:
            feature_dict['aatype'] = jnp.asarray(feature_dict['aatype'])
        if 'msa' in feature_dict:
             # Ensure MSA is float32 if it exists
            if feature_dict['msa'] is not None:
                 feature_dict['msa'] = jnp.asarray(feature_dict['msa'], dtype=jnp.float32)
        logging.debug("Converted 'aatype' and 'msa' in feature_dict to JAX arrays.")

        # Delegate to the selected protocol instance
        design_results, final_feature_dict = self.protocol_instance.design(
            fold_input, feature_dict, target_indices, binder_indices, rng_key
        )
        return design_results, final_feature_dict

    # --- run_final_prediction and run_complete_prediction are now handled by protocols ---
    # They can be called on self.protocol_instance if needed externally

# Keep the top-level design_binder function as the main entry point
# It now uses the BinderDesigner factory

def design_binder(
    fold_input: folding_input.Input,
    feature_dict: features.BatchDict,
    model_runner: Any,
    ccd: chemical_components.Ccd,
    design_config: DesignConfig,
    rng_seed: int = 0,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    use_complete_prediction: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], model.ModelResult, features.BatchDict, folding_input.Input]:
    """Design a binder protein using AlphaFold 3.
    
    Args:
        fold_input: The input to AlphaFold.
        feature_dict: The feature dictionary for the model.
        model_runner: ModelRunner instance.
        ccd: Chemical component dictionary.
        design_config: Structured design configuration.
        rng_seed: Random seed for JAX.
        buckets: Optional bucket sizes for featurization of final prediction.
        ref_max_modified_date: Optional reference date for chemical components.
        conformer_max_iterations: Optional iterations for conformer generation.
        use_complete_prediction: Whether to use the comprehensive prediction method
                                 or the simplified one for the final structure.
        output_dir: Optional path to the output directory for saving plots.
        
    Returns:
        Tuple of (design_results, final_model_result, final_feature_dict, new_fold_input).
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    design_key, final_pred_key = jax.random.split(rng_key)
    
    # Initialize designer factory
    designer = BinderDesigner(model_runner, ccd, design_config)
    
    # Run design using the selected protocol
    design_results, best_feature_dict = designer.design_binder(
        fold_input, feature_dict, design_key
    )

    # Store necessary results from design for final prediction
    # The protocol itself should store these in the design_results dict
    design_results["best_feature_dict"] = best_feature_dict # Ensure this is passed back
    if "binder_indices" not in design_results:
        design_results["binder_indices"] = designer._binder_indices
    if "target_indices" not in design_results:
        design_results["target_indices"] = designer._target_indices
    # final_seq_logits should also be in design_results from the protocol

    # Run final prediction using the chosen method on the protocol instance
    if use_complete_prediction:
        logging.info("Running complete final prediction...")
        final_model_result, final_feature_dict, new_fold_input = designer.protocol_instance.run_complete_prediction(
            fold_input, 
            design_results, 
            final_pred_key,
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations
        )
    else:
        logging.info("Running simplified final prediction...")
        final_model_result, final_feature_dict, new_fold_input = designer.protocol_instance.run_final_prediction(
            fold_input, design_results, final_pred_key
        )
    
    # --- Plotting Trajectory --- >
    if output_dir and "trajectory" in design_results:
        try:
            plot_filename_prefix = f"{fold_input.name}_seed_{rng_seed}_design"
            plotting.plot_design_trajectory(
                design_results=design_results,
                output_dir=output_dir,
                plot_filename_prefix=plot_filename_prefix,
            )
        except Exception as e:
            logging.error(f"Failed to plot design trajectory: {e}", exc_info=True)
    # <--- End Plotting ---
    
    return design_results, final_model_result, final_feature_dict, new_fold_input 

# --- Removed utility functions (moved to utils.py) --- 
# safe_jax_to_float, safe_process_losses, freeze_containers_for_jax

# --- Removed internal protocol methods --- 
# _design_binder_gradient, _design_binder_boltz, _boltz_forward_pass

# --- Removed final prediction methods (moved to protocols) ---
# run_final_prediction, run_complete_prediction 
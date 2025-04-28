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
from alphafold3.design.config import DesignConfig, ALPHABET_SIZE
from alphafold3.design import plotting
from alphafold3.design import sequence_utils

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
        # Add initial sequence and bias options for STE
        initial_binder_sequence: Optional[str] = None,
        initial_bias: Optional[np.ndarray] = None,
        # TODO: Add options for initialization mode (random, wildtype)
        # initialization_mode: str = "random",
        # random_init_scale: float = 0.01, # Could get from config
    ):
        """Initialize the binder designer factory with a DesignConfig.
        
        Args:
            model_runner: ModelRunner instance.
            ccd: Chemical component dictionary.
            design_config: Structured design configuration.
            initial_binder_sequence: Optional initial sequence for the binder.
            initial_bias: Optional initial bias array (L, ALPHABET_SIZE).
        """
        self.model_runner = model_runner
        self.ccd = ccd
        self.config = design_config
        self.protocol_name = self.config.protocol_name

        # Initialize dictionaries for parameters and inputs (similar to ColabDesign)
        self.params: Dict[str, jnp.ndarray] = {}
        self.inputs: Dict[str, Any] = {}
        self.optimizer: Optional[optax.GradientTransformation] = None
        self.optimizer_state: Optional[optax.OptState] = None

        # Populate self.opt with STE parameters from the specific protocol config
        self.opt: Dict[str, Any] = {}
        protocol_cfg = self.config.gradient_config if self.config.protocol_name == "binder_gradient" else self.config.boltz_config
        if protocol_cfg:
            # Extract STE params - use vars() for simplicity
            cfg_vars = vars(protocol_cfg)
            ste_params = {k: v for k, v in cfg_vars.items() if k.startswith("ste_")}
            self.opt.update(ste_params)

        # Store initial values if provided, bias needs shape validation later
        self._initial_binder_sequence = initial_binder_sequence
        self._initial_bias = initial_bias

        # Amino acid mapping for wildtype init
        self._aa_order = 'ACDEFGHIKLMNPQRSTVWY'
        self._aa_map = {aa: i for i, aa in enumerate(self._aa_order)}

        # Create protocol instance AFTER basic setup
        self.protocol_instance = self._create_protocol()

        # Set optimizer based on config AFTER protocol instance is created (in case it needs it)
        self.set_optimizer(optimizer_name=self.config.optimizer_name)

        logging.info(f"Initialized BinderDesigner with protocol: {self.protocol_name}")
        logging.debug(f"Design config: {self.config}")

    def _initialize_logits_and_bias(self, binder_length: int, rng_key: jnp.ndarray):
        """Initializes seq_logits and bias based on provided values or defaults."""
        # Initialize logits
        if "seq_logits" not in self.params:
            logits_shape = (binder_length, ALPHABET_SIZE)
            if self._initial_binder_sequence is not None:
                logging.info(f"Initializing seq_logits from provided sequence: {self._initial_binder_sequence[:10]}...")
                if len(self._initial_binder_sequence) != binder_length:
                    raise ValueError(
                        f"Length mismatch: initial_binder_sequence ({len(self._initial_binder_sequence)}) "
                        f"!= binder_length ({binder_length})"
                    )
                # Convert sequence to one-hot
                try:
                    indices = [self._aa_map[aa.upper()] for aa in self._initial_binder_sequence]
                    one_hot = jax.nn.one_hot(jnp.array(indices), num_classes=ALPHABET_SIZE)
                    # Initialize logits to be high for the target AA, low otherwise
                    # (e.g., 1.0 for target, -1.0 for others, scale later if needed)
                    self.params['seq_logits'] = (one_hot * 2.0 - 1.0) * 5.0 # Strong initial bias
                except KeyError as e:
                    raise ValueError(f"Invalid character '{e}' in initial_binder_sequence.")
            else:
                # Use scale from config if available, else default
                random_init_scale = 0.01 # Default scale
                if self.config.boltz_config: # Check boltz first as it has the scale
                    random_init_scale = self.config.boltz_config.random_init_scale
                elif self.config.gradient_config:
                     # Gradient config doesn't currently have this, use default
                     pass
                logging.info(f"Initializing random seq_logits with scale {random_init_scale}")
                key, subkey = jax.random.split(rng_key)
                self.params['seq_logits'] = random_init_scale * jax.random.normal(
                    subkey, shape=logits_shape
                )

            self.params['seq_logits'] = jnp.asarray(self.params['seq_logits'], dtype=jnp.float32)
        else:
             logging.info("Using existing seq_logits.")

        # Initialize bias
        if "bias" not in self.inputs:
            if self._initial_bias is not None:
                if self._initial_bias.shape == (binder_length, ALPHABET_SIZE):
                    self.inputs['bias'] = jnp.asarray(self._initial_bias, dtype=jnp.float32)
                    logging.info("Initialized bias from provided array.")
                else:
                    logging.warning(f"Provided initial_bias shape {self._initial_bias.shape} doesn't match expected ({binder_length}, {ALPHABET_SIZE}). Using zero bias.")
                    self.inputs['bias'] = jnp.zeros((binder_length, ALPHABET_SIZE), dtype=jnp.float32)
            else:
                logging.info("Initializing bias to zeros.")
                self.inputs['bias'] = jnp.zeros((binder_length, ALPHABET_SIZE), dtype=jnp.float32)
        else:
            logging.info("Using existing bias.")

    def set_bias(self, bias: np.ndarray):
        """Sets the sequence bias array, useful for excluding amino acids."""
        # Add shape validation if binder length is known
        # binder_len = self.params.get('seq_logits', {}).get('shape', [None])[0]
        # if binder_len is not None and bias.shape[0] != binder_len:
        #    raise ValueError("Bias length does not match binder length")
        self.inputs['bias'] = jnp.asarray(bias, dtype=jnp.float32)
        logging.info(f"Set sequence bias with shape {self.inputs['bias'].shape}")

    def set_optimizer(self, optimizer_name: str = "adam", learning_rate: Optional[float] = None):
        """Sets the Optax optimizer."""
        lr = learning_rate if learning_rate is not None else self.config.learning_rate
        if optimizer_name.lower() == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optax.adamw(learning_rate=lr)
        # Add other optimizers if needed (e.g., sgd)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Initialize optimizer state if params exist
        if self.params:
            self.optimizer_state = self.optimizer.init(self.params)
            logging.info(f"Initialized {optimizer_name} optimizer with LR={lr} and state.")
        else:
            logging.info(f"Set {optimizer_name} optimizer with LR={lr}. State will be initialized later.")

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
        # Pass optimization settings
        opt: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """Design a binder protein using the selected protocol.
        
        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The feature dictionary for the model.
            rng_key: JAX random key.
            opt: Dictionary with optimization settings (temp, alpha, soft, hard).
            
        Returns:
            Tuple of (design_results, final_feature_dict).
        """
        key_setup, key_design = jax.random.split(rng_key)

        # Store optimization settings if provided
        if opt:
             self.opt.update(opt)

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
        binder_length = len(binder_indices)

        # *** Initialize sequence logits and bias HERE ***
        # Needs binder_length, which is determined after setup_binder_features
        self._initialize_logits_and_bias(binder_length, key_setup)

        # *** Initialize Optimizer State HERE ***
        # Requires self.params to be initialized
        if self.optimizer and not self.optimizer_state:
             if self.params:
                  self.optimizer_state = self.optimizer.init(self.params)
                  logging.info("Initialized optimizer state.")
             else:
                  logging.warning("Optimizer set, but no params available to initialize state.")

        # Convert necessary numpy arrays in feature_dict to JAX arrays for gradient updates
        # This might be better handled within the protocols if needed differently
        if 'aatype' in feature_dict:
            feature_dict['aatype'] = jnp.asarray(feature_dict['aatype'])
        if 'msa' in feature_dict:
             # Ensure MSA is float32 if it exists
            if feature_dict['msa'] is not None:
                 feature_dict['msa'] = jnp.asarray(feature_dict['msa'], dtype=jnp.float32)
        logging.debug("Converted 'aatype' and 'msa' in feature_dict to JAX arrays.")

        # Pass necessary state to the protocol's design method
        # The protocol will be responsible for using params, inputs, opt, etc.
        design_results, final_feature_dict = self.protocol_instance.design(
            fold_input=fold_input,
            feature_dict=feature_dict,
            target_indices=target_indices,
            binder_indices=binder_indices,
            rng_key=key_design,
            designer_params=self.params, # Pass trainable params
            designer_inputs=self.inputs, # Pass bias etc.
            designer_opt=self.opt,       # Pass optimization settings
            optimizer=self.optimizer,
            optimizer_state=self.optimizer_state
        )

        # Update state after protocol run (optimizer state might change)
        if 'optimizer_state' in design_results:
            self.optimizer_state = design_results.pop('optimizer_state')
        if 'params' in design_results:
             self.params = design_results.pop('params') # Update params if protocol modifies them

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
    # Add STE-related options for top-level call
    initial_binder_sequence: Optional[str] = None,
    initial_bias: Optional[np.ndarray] = None,
    # TODO: Add flags for opt dict (temp, alpha, soft, hard)
    design_opt: Optional[Dict[str, Any]] = None,
    optimizer_name: str = "adam", # Default optimizer
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
        initial_binder_sequence: Optional initial sequence for the binder.
        initial_bias: Optional initial bias array (L, ALPHABET_SIZE).
        design_opt: Optional dictionary with design optimization settings.
        optimizer_name: Optional name of the optimizer to use.
        
    Returns:
        Tuple of (design_results, final_model_result, final_feature_dict, new_fold_input).
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    design_key, final_pred_key = jax.random.split(rng_key)
    
    # Initialize designer factory, passing STE options
    designer = BinderDesigner(
        model_runner, ccd, design_config,
        initial_binder_sequence=initial_binder_sequence,
        initial_bias=initial_bias
    )

    # Set the optimizer on the designer instance
    designer.set_optimizer(optimizer_name=optimizer_name)

    # Run design using the selected protocol, passing optimization settings
    # The design_binder method now handles initializing logits/bias and optimizer state
    design_results, best_feature_dict = designer.design_binder(
        fold_input, feature_dict, design_key, opt=design_opt
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
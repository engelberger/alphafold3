"""Model components for BoltzDesign1 protocol."""

import functools
import haiku as hk
import jax
import jax.numpy as jnp
from absl import logging
from typing import Dict, Any, Callable

from alphafold3.model import model as af3_model_lib
from alphafold3.model import model_config
from alphafold3.model import features
from alphafold3.model import feat_batch
from alphafold3.model.components import utils
from alphafold3.model.components import mapping
from alphafold3.model.network import featurization
from alphafold3.model.network import distogram_head
from alphafold3.model.network import confidence_head
from alphafold3.model.network import diffusion_head
from alphafold3.model.network import evoformer as evoformer_network


class BoltzDesignModel(hk.Module):
    """A Haiku module for the modified BoltzDesign1 forward pass."""

    def __init__(self, config: af3_model_lib.Model.Config, name: str = "boltz_design_model"):
        super().__init__(name=name)
        # Store config needed by sub-modules
        self.config = config
        self.global_config = config.global_config

        # Instantiate necessary sub-modules
        self.embedding_module = evoformer_network.Evoformer(
            self.config.evoformer, self.global_config
        )
        self.structure_module = af3_model_lib.diffusion_head.DiffusionHead(
            self.config.heads.diffusion, self.global_config
        )
        self.confidence_head = confidence_head.ConfidenceHead(
            self.config.heads.confidence, self.global_config
        )
        self.distogram_head = distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config
        )

    def __call__(self, batch: features.BatchDict, rng_key: jnp.ndarray) -> Dict[str, Any]:
        """Performs the BoltzDesign1 forward pass."""
        if isinstance(batch, dict):
            batch = feat_batch.Batch.from_data_dict(batch)

        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        # --- 0. Initial feature processing ---
        logging.debug("Preparing initial representations for Boltz forward pass...")
        target_feat = af3_model_lib.create_target_feat_embedding(
            batch=batch,
            config=self.embedding_module.config,
            global_config=self.global_config,
        )

        # --- 1. Run Pairformer (Evoformer) ---
        logging.debug("Running Pairformer...")
        # Initialize embeddings for recycling
        num_res = batch.num_res
        prev_embeddings = {
            'pair': jnp.zeros(
                [num_res, num_res, self.config.evoformer.pair_channel],
                dtype=jnp.float32,
            ),
            'single': jnp.zeros(
                [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
            ),
            'target_feat': target_feat,
        }

        # Run Evoformer with recycling
        embeddings = prev_embeddings
        for i in range(self.config.num_recycles + 1):
            recycling_key, key1 = jax.random.split(key1)
            embeddings = self.embedding_module(
                batch=batch,
                prev=embeddings,
                target_feat=target_feat,
                key=recycling_key,
            )
            # Convert to float32 for numerical stability
            embeddings['pair'] = embeddings['pair'].astype(jnp.float32)
            embeddings['single'] = embeddings['single'].astype(jnp.float32)
            
        # --- 2. Generate Distogram ---
        logging.debug("Computing distogram...")
        distogram_result = self.distogram_head(batch, embeddings)

        # --- 3. Run Structure Module (with stop_gradient) ---
        logging.debug("Running Structure Module (with stop_gradient)...")
        
        # Config for diffusion sampling
        sample_config = self.config.heads.diffusion.eval
        
        # Partial function for structure module
        denoising_step = functools.partial(
            self.structure_module,
            batch=batch,
            embeddings=embeddings,
            use_conditioning=True,
        )

        # Run structure module for diffusion sampling
        structure_result = af3_model_lib.diffusion_head.sample(
            denoising_step=denoising_step,
            batch=batch, 
            key=key2,
            config=sample_config,
        )
        
        # Get final atom positions and apply stop_gradient
        atom_positions = structure_result['atom_positions']
        atom_positions_no_grad = jax.lax.stop_gradient(atom_positions)
        
        # --- 4. Run Confidence Head ---
        logging.debug("Running Confidence Head...")
        # Map the confidence head over all diffusion samples
        # but with no gradient through atom_positions
        confidence_output = af3_model_lib.mapping.sharded_map(
            lambda dense_atom_positions: self.confidence_head(
                dense_atom_positions=dense_atom_positions,
                embeddings=embeddings,
                seq_mask=batch.token_features.mask,
                token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch.token_features.asym_id,
            ),
            in_axes=0,
        )(atom_positions_no_grad)
        
        # --- 5. Combine results ---
        boltz_result = {
            "distogram": distogram_result,
            "confidence_outputs": confidence_output,
            "final_atom_positions_no_grad": atom_positions_no_grad,
            "embeddings": embeddings,
        }
        
        logging.debug("Boltz forward pass complete.")
        return boltz_result


def run_boltz_forward_pass(
    model_params: hk.Params,
    config: af3_model_lib.Model.Config,
    feature_dict: Dict[str, jnp.ndarray],
    rng_key: jnp.ndarray
) -> Dict[str, Any]:
    """Transforms and applies the BoltzDesignModel."""
    
    def _forward_fn(batch):
        # Instantiate the BoltzDesignModel inside the transformed function
        return BoltzDesignModel(config=config)(batch, rng_key=rng_key)

    # Transform the function
    transformed_forward = hk.transform(_forward_fn)
    
    # Apply the transformed function with model parameters and input features
    return transformed_forward.apply(model_params, rng=None, batch=feature_dict)


# For testing/debugging
if __name__ == "__main__":
    import os
    import sys
    import time
    from absl import app
    
    def main(argv):
        # Set up basic logging
        logging.set_verbosity(logging.INFO)
        logging.info("Testing BoltzDesignModel implementation...")
        
        try:
            # Simple diagnostic to check the environment
            logging.info(f"Current directory: {os.getcwd()}")
            logging.info(f"Python path: {sys.path}")
            
            # Load a model runner
            logging.info("Loading ModelRunner...")
            from alphafold3.model import model
            from alphafold3.model.model_config import GlobalConfig
            
            # Create minimal configs
            logging.info("Creating minimal model configs for testing...")
            global_config = GlobalConfig()
            model_config = model.Model.Config(
                global_config=global_config,
                heads=model.Model.HeadsConfig(),
                evoformer=evoformer_network.Evoformer.Config(),
                num_recycles=1  # Use fewer recycles for testing
            )
            
            logging.info("BoltzDesignModel can be imported successfully")
            logging.info("This is a place where you would test the implementation")
            logging.info("To fully test the model, you need real model parameters and feature dictionaries")
            logging.info("Test complete")
        
        except Exception as e:
            logging.error(f"Error occurred during testing: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the test function
    app.run(main) 
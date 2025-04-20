"""Base class for binder design protocols."""

import abc
import datetime
import logging
from typing import Dict, Any, Tuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import model
from alphafold3.model import features
from alphafold3.design import binder_utils


class BinderProtocol(abc.ABC):
    """Abstract base class for all binder design protocols."""

    def __init__(self, model_runner: Any, ccd: chemical_components.Ccd, design_params: Dict[str, Any]):
        self.model_runner = model_runner
        self.ccd = ccd
        self.design_params = design_params
        self.protocol = design_params.get("protocol", "unknown")
        self.clear_memory_interval = design_params.get("clear_memory_interval", 0)

    @abc.abstractmethod
    def design(
        self,
        fold_input: folding_input.Input,
        feature_dict: features.BatchDict,
        target_indices: np.ndarray,
        binder_indices: np.ndarray,
        rng_key: jnp.ndarray,
    ) -> Tuple[Dict[str, Any], features.BatchDict]:
        """Main design method to be implemented by concrete protocols.

        Args:
            fold_input: The input to AlphaFold.
            feature_dict: The feature dictionary for the model.
            target_indices: Indices of target residues.
            binder_indices: Indices of binder residues.
            rng_key: JAX random key.

        Returns:
            Tuple of (design_results, final_feature_dict).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_final_prediction(
        self,
        fold_input: folding_input.Input,
        design_results: Dict[str, Any],
        rng_key: jnp.ndarray,
    ) -> Tuple[model.ModelResult, features.BatchDict, folding_input.Input]:
        """Run simplified final prediction with the designed sequence.

        Args:
            fold_input: The original input to AlphaFold.
            design_results: Results from the design process.
            rng_key: JAX random key.

        Returns:
            Tuple of (model_result, final_feature_dict, new_fold_input).
        """
        raise NotImplementedError

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
        if "final_aa_indices" not in design_results:
            logging.error("final_aa_indices not found in design_results. Cannot run complete prediction.")
            # Return empty/placeholder results or raise an error
            return {}, {}, fold_input

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
        binder_chains = self.design_params.get("binder_chains", [])
        if not binder_chains:
            logging.error("Binder chains not specified in design_params. Cannot create new input.")
            return {}, {}, fold_input

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
                if binder_start_idx + chain_length > len(designed_sequence):
                    logging.error(f"Designed sequence length ({len(designed_sequence)}) insufficient for binder chain {binder_chain_id} (length {chain_length}) starting at index {binder_start_idx}.")
                    return {}, {}, fold_input
                binder_seq_by_chain[binder_chain_id] = designed_sequence[binder_start_idx:binder_start_idx + chain_length]
                binder_start_idx += chain_length
            else:
                logging.warning(f"Binder chain ID {binder_chain_id} specified in design_params not found in original input chains.")

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
                    # This can happen if binder chain ID was in params but not in input
                    logging.warning(f"No designed sequence for binder chain {chain.id}, using original")
                    new_chains.append(chain)
            else:
                # This is a target chain or non-protein chain - keep it unchanged
                new_chains.append(chain)

        # Create a new Input object with the updated chains
        new_fold_input = folding_input.Input(
            name=fold_input.name + "_designed",
            chains=new_chains,
            rng_seeds=[fold_input.rng_seeds[0]] if fold_input.rng_seeds else [0], # Use only the first seed or default
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
            if not featurised_examples:
                 logging.error("Featurization returned empty results. Cannot run inference.")
                 return {}, {}, new_fold_input
            final_feature_dict = featurised_examples[0]

            # Run standard prediction
            logging.info("Running inference on completely featurized designed sequence...")
            final_result = self.model_runner.run_inference(final_feature_dict, rng_key)

            logging.info("Complete prediction finished successfully")

            # Add designed sequence metadata
            if isinstance(final_result, dict):
                 if "metadata" not in final_result:
                     final_result["metadata"] = {}
                 final_result["metadata"]["designed_sequence"] = designed_sequence
                 final_result["metadata"]["designed_binder_chains"] = binder_chains
            else:
                 # Handle case where model_runner might not return a dict
                 logging.warning(f"Final model result is not a dict: {type(final_result)}. Cannot add metadata.")
                 final_result = {"output": final_result, "metadata": {"designed_sequence": designed_sequence, "designed_binder_chains": binder_chains}}

            return final_result, final_feature_dict, new_fold_input

        except ImportError as e:
            logging.warning(f"Could not run complete prediction due to missing import: {str(e)}")
            logging.warning("Falling back to simplified prediction method")

            # Fall back to the simpler approach
            final_result_simple, final_feature_dict_simple, new_fold_input_simple = self.run_final_prediction(fold_input, design_results, rng_key)
            return final_result_simple, final_feature_dict_simple, new_fold_input_simple
        except Exception as e:
            logging.error(f"Error during complete prediction: {e}", exc_info=True)
            # Return empty/placeholder results
            return {}, {}, new_fold_input 
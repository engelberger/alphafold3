# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import enum
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import overload, Optional, Dict, Any
import copy

from absl import app
from absl import flags
from absl import logging
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
from alphafold3.data.custom_utils import parse_mutation_string, apply_mutations_to_input, parse_masking_positions, apply_masking_to_features
from alphafold3.model import data_constants
from alphafold3.design import binder_design

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

# Template search configuration.
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All'
    ' templates released after this date will be ignored. Controls also whether'
    ' to allow use of model coordinates for a chemical component from the CCD'
    ' if RDKit conformer generation fails and the component does not have ideal'
    ' coordinates set. Only for components that have been released before this'
    ' date the model coordinates can be used as a fallback.',
)

_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference. Defaults to the'
    ' 1st GPU on the system. Useful on multi-GPU systems to pin each run to a'
    ' specific GPU.',
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Protocol selection and binder design parameters
class DesignProtocol(enum.Enum):
    FOLD = 'fold'                 # Standard prediction
    BINDER_GRADIENT = 'binder_gradient' # ColabDesign-like approach
    BINDER_BOLTZ = 'binder_boltz'     # BoltzDesign1 approach

flags.DEFINE_enum_class(
    'protocol', DesignProtocol.FOLD, DesignProtocol,
    'The prediction or design protocol to run.'
)
_TARGET_CHAINS = flags.DEFINE_list(
    'target_chains', None,
    'Comma-separated list of chain IDs to keep fixed (target). Required for binder protocols.'
)
_BINDER_CHAINS = flags.DEFINE_list(
    'binder_chains', None,
    'Comma-separated list of chain IDs to design (binder). Required for binder protocols.'
)

# General Design Parameters
_DESIGN_LEARNING_RATE = flags.DEFINE_float(
    'design_learning_rate', 0.1, 'Base learning rate for sequence optimization.'
)
_DESIGN_STEPS = flags.DEFINE_integer(
    'design_steps', 200, 'Total optimization steps (used if protocol has one stage).'
)
_DESIGN_SEQ_ENTROPY_WEIGHT = flags.DEFINE_float(
    'design_seq_entropy_weight', 0.01, 'Weight for sequence entropy loss.'
)

_CLEAR_MEMORY_INTERVAL = flags.DEFINE_integer(
    'clear_memory_interval', 
    0, 
    'Interval (in steps) at which to clear GPU/CPU memory during binder design. '
    'Set to 0 to disable memory clearing. '
    'Values between 10-50 are recommended for GPUs with limited memory.'
)

# Gradient-Specific Parameters
_GRADIENT_PLDDT_WEIGHT = flags.DEFINE_float(
    'gradient_plddt_weight', 0.5, 'Weight for binder pLDDT loss (gradient protocol).'
)
_GRADIENT_PAE_INTER_WEIGHT = flags.DEFINE_float(
    'gradient_pae_inter_weight', 0.5, 'Weight for interface PAE loss (gradient protocol).'
)
_GRADIENT_CONTACT_WEIGHT = flags.DEFINE_float(
    'gradient_contact_weight', 0.5, 'Weight for interface contact loss (gradient protocol).'
)

# Boltz-Specific Parameters
_BOLTZ_STAGE1_STEPS = flags.DEFINE_integer(
    'boltz_stage1_steps', 50, 'Steps for BoltzDesign1 Stage 1 (exploration).'
)
_BOLTZ_STAGE2_STEPS = flags.DEFINE_integer(
    'boltz_stage2_steps', 50, 'Steps for BoltzDesign1 Stage 2 (transition).'
)
_BOLTZ_STAGE3_STEPS = flags.DEFINE_integer(
    'boltz_stage3_steps', 50, 'Steps for BoltzDesign1 Stage 3 (convergence).'
)
_BOLTZ_STAGE4_STEPS = flags.DEFINE_integer(
    'boltz_stage4_steps', 50, 'Steps for BoltzDesign1 Stage 4 (one-hot).'
)
_BOLTZ_CONTACT_INTRA_WEIGHT = flags.DEFINE_float(
    'boltz_contact_intra_weight', 1.0, 'Weight for intra-binder distogram entropy loss (Boltz protocol).'
)
_BOLTZ_CONTACT_INTER_WEIGHT = flags.DEFINE_float(
    'boltz_contact_inter_weight', 1.0, 'Weight for inter-face distogram entropy loss (Boltz protocol).'
)
_BOLTZ_CONFIDENCE_WEIGHT = flags.DEFINE_float(
    'boltz_confidence_weight', 0.5, 'Weight for confidence loss (pLDDT/PAE) (Boltz protocol).'
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.',
)
_FORCE_OUTPUT_DIR = flags.DEFINE_bool(
    'force_output_dir',
    False,
    'Whether to force the output directory to be used even if it already exists'
    ' and is non-empty. Useful to set this to True to run the data pipeline and'
    ' the inference separately, but use the same output directory.',
)

# --- Mutation and Masking Flags ---
flags.DEFINE_string('mutations', None,
                    'Comma-separated list of mutations, e.g., "A123G,R45C" or '
                    '"A:G10C,B:R20D". Assumes 1-based indexing.')
flags.DEFINE_string('mask_positions', None,
                    'Comma-separated list and/or ranges of residue positions to '
                    'mask in MSA/deletion matrix, e.g., "10-15,20,30". '
                    'Assumes 1-based indexing.')
flags.DEFINE_boolean('mask_msa', False,
                     'Enable masking of specified positions in the MSA feature.')
flags.DEFINE_boolean('mask_deletion_matrix', False,
                     'Enable masking (zeroing) of specified positions in the '
                     'deletion matrix.')

# Attempt to get the list for validation, provide a fallback
try:
    # Ensure data_constants is imported and has this attribute
    _VALID_MASK_TOKENS = data_constants.protein_restypes_with_unk_and_gap
except (AttributeError, NameError):
     logging.warning("Could not load valid mask tokens from data_constants. Using default list.")
     _VALID_MASK_TOKENS = list('ACDEFGHIKLMNPQRSTVWYX-')

flags.DEFINE_enum('mask_token', 'X', _VALID_MASK_TOKENS,
                  'Amino acid character to use for masking MSA positions.')


# --- Logging and Metrics ---
_LOG_LEVEL = flags.DEFINE_string(
    'log_level',
    'INFO',
    'Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
)
_LOG_FORMAT = flags.DEFINE_enum(
    'log_format',
    'pretty',
    ['pretty', 'json', 'tsv'],
    'Format for logs (pretty=human readable, json=structured JSON, tsv=tab separated)',
)
_LOG_FILE = flags.DEFINE_string(
    'log_file',
    None,
    'Optional file path to write logs to (in addition to console)',
)
_LOG_CONFIG_FILE = flags.DEFINE_string(
    'log_config_file',
    None,
    'Optional JSON or YAML file with detailed logging configuration',
)
_METRICS_FORMAT = flags.DEFINE_enum(
    'metrics_format',
    'json',
    ['json', 'csv'],
    'Format for metrics files (json=JSONL, csv=comma separated)',
)
_METRICS_WRITE_INTERVAL = flags.DEFINE_integer(
    'metrics_write_interval',
    10,
    'How often to write metrics files (every N steps)',
)

# set TF_FORCE_UNIFIED_MEMORY=true and XLA_PYTHON_CLIENT_MEM_FRACTION=3.2
#os.environ["TF_FORCE_UNIFIED_MEMORY"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "3.2"

# ----------------------------------

FLAGS = flags.FLAGS # Define FLAGS after all flags are defined

# --- Add new flags ---
flags.DEFINE_integer(
    'binder_length',
    None,
    'If set, overrides the length of the specified binder chains with this '
    'value, initializing the sequence with Alanines (\'A\').',
)
flags.DEFINE_integer(
    'seed',
    None,
    'If set, overrides the random seeds specified in the input JSON with this '
    'single seed.',
)
# --------------------

def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = model_dir

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict, str | None], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch, mode=None):
      return model.Model(self._model_config)(batch, mode=mode)

    return functools.partial(
        jax.jit(forward_fn.apply, static_argnames=["mode"], device=self._device), self.model_params
    )

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray, mode: str | None = None,
      # Add optional override for number of recycles
      num_recycles_override: Optional[int] = None
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example.

    Args:
        featurised_example: Featurized input data.
        rng_key: JAX random key.
        mode: Optional mode to run the model in. Options:
            None (default): Standard prediction mode.
            "boltz_design": BoltzDesign1 mode (may trigger modifications like recycles=0).
        num_recycles_override: If set, overrides the default number of recycles
                                 from the model config.

    Returns:
        Model results from the forward pass.
    """
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    # Prepare a copy of the config to potentially modify
    current_config = copy.deepcopy(self._model_config)

    # Override recycles if specified or if in boltz_design mode (set to 0)
    if num_recycles_override is not None:
        current_config.num_recycles = num_recycles_override
        logging.debug(f"Overriding num_recycles to {num_recycles_override}")
    elif mode == "boltz_design":
        original_recycles = current_config.num_recycles
        current_config.num_recycles = 0
        logging.debug(f"Setting num_recycles to 0 for boltz_design mode (original: {original_recycles})")

    # --- Dynamically create the forward function with potentially modified config --- 
    # This ensures the correct config (e.g., num_recycles=0) is used for this specific call
    # We avoid caching this modified forward function to prevent conflicts
    @hk.transform
    def modified_forward_fn(batch, mode_inner=None):
        # Use the potentially modified current_config
        return model.Model(current_config)(batch, mode=mode_inner)

    # JIT compile the modified function for this call
    # Note: This recompiles if config changes, which is expected for boltz_design mode
    jitted_modified_model = jax.jit(modified_forward_fn.apply, static_argnames=["mode_inner"], device=self._device)
    result = jitted_modified_model(self.model_params, rng_key, featurised_example, mode_inner=mode)
    # --------------------------------------------------------------------------------

    # Apply stop_gradient to the entire result dictionary for boltz_design mode.
    # NOTE: This is a coarse application. Ideally, stop_gradient should be applied
    # more selectively, e.g., only to structure module outputs before they are used
    # by the confidence head (if confidence is calculated separately).
    # A more refined approach might require modifying the ModelRunner or Model itself.
    if mode == "boltz_design":
        logging.debug("Applying stop_gradient to the full result in boltz_design mode (Coarse Implementation)")
        result = jax.lax.stop_gradient(result)

    # Skip NumPy conversion when in boltz_design mode to avoid TracerArrayConversionError
    # This is needed because boltz_design mode runs within a JAX JIT context
    if mode != "boltz_design":
      # Only convert to numpy arrays when not in boltz_design mode
      result = jax.tree.map(np.asarray, result)
      result = jax.tree.map(
          lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
          result,
      )
    
    result = dict(result)
    
    # Add identifier only in standard mode, not in boltz_design mode
    # since bytes values cause errors in JIT contexts
    if mode != "boltz_design":
      identifier = self.model_params['__meta__']['__identifier__'].tobytes()
      result['__identifier__'] = identifier
    
    return result

  def extract_inference_results_and_maybe_embeddings(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> tuple[list[model.InferenceResult], dict[str, np.ndarray] | None]:
    """Extracts inference results and embeddings (if set) from model outputs."""
    inference_results = list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )
    num_tokens = len(inference_results[0].metadata['token_chain_ids'])
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][:num_tokens]
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ]
    return inference_results, embeddings or None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    masking_config: Optional[Dict[str, Any]] = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
  featurisation_start_time = time.time()
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      verbose=True,
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations,
      masking_config=masking_config,
  )
  print(
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference with seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key)
    print(
        f'Running model inference with seed {seed} took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting inference results with seed {seed}...')
    extract_structures = time.time()
    inference_results, embeddings = (
        model_runner.extract_inference_results_and_maybe_embeddings(
            batch=example, result=result, target_name=fold_input.name
        )
    )
    print(
        f'Extracting {len(inference_results)} inference samples with'
        f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
    )

    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result,
          output_dir=sample_dir,
          name=f'{job_name}_seed-{seed}_sample-{sample_idx}',
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings,
          output_dir=embeddings_dir,
          name=f'{job_name}_seed-{seed}',
      )

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(
        os.path.join(output_dir, f'{job_name}_ranking_scores.csv'), 'wt'
    ) as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    mutations_str: Optional[str],
    masking_config: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    force_output_dir: bool = False,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    mutations_str: Optional[str],
    masking_config: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    force_output_dir: bool = False,
) -> Sequence[ResultsForSeed]:
  ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    mutations_str: Optional[str],
    masking_config: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    force_output_dir: bool = False,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    mutations_str: Optional string defining mutations to apply.
    masking_config: Dictionary with masking parameters.
    design_params: Optional dictionary with design parameters.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    ref_max_modified_date: Optional maximum date that controls whether to allow
      use of model coordinates for a chemical component from the CCD if RDKit
      conformer generation fails and the component does not have ideal
      coordinates set. Only for components that have been released before this
      date the model coordinates can be used as a fallback.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    force_output_dir: If True, do not create a new output directory even if the
      existing one is non-empty. Instead use the existing output directory and
      potentially overwrite existing files. If False, create a new timestamped
      output directory instead if the existing one is non-empty.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'\nRunning fold job {fold_input.name}...')
  logging.info(f"Processing input: {fold_input.name}")

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if (
      not force_output_dir
      and os.path.exists(output_dir)
      and os.listdir(output_dir)
  ):
    new_output_dir = (
        f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(
        f'Output will be written in {new_output_dir} since {output_dir} is'
        ' non-empty.'
    )
    logging.warning(f"Output directory '{output_dir}' is non-empty. Using '{new_output_dir}' instead.")
    output_dir = new_output_dir
  else:
    print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
    logging.info("Skipping data pipeline as config is None.")
  else:
    print('Running data pipeline...')
    logging.info("Running data pipeline...")
    try:
        data_pipeline_runner = pipeline.DataPipeline(
            data_pipeline_config=data_pipeline_config,
            mutations_str=mutations_str,
            masking_config=masking_config
        )
        fold_input = data_pipeline_runner.process(fold_input)
        logging.info("Data pipeline finished successfully.")
    except Exception as e:
        logging.error(f"Data pipeline failed for job {fold_input.name}: {e}", exc_info=True)
        raise

  write_fold_input_json(fold_input, output_dir)

  if model_runner is None:
    print('Skipping model inference...')
    logging.info("Skipping model inference as model_runner is None.")
    output = fold_input
  else:
    # Check if we're running a binder design protocol
    if design_params is not None and design_params.get("protocol") in ["binder_gradient", "binder_boltz"]:
      print(f'Designing binder for {fold_input.name} using {design_params["protocol"]} protocol...')
      logging.info(f"Starting binder design for {fold_input.name} with protocol {design_params['protocol']}")
      
      # Get CCD
      ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
      
      # Featurize the input (same as for prediction)
      # This will create features for *each* seed specified in the input
      featurisation_start_time = time.time()
      featurised_examples = featurisation.featurise_input(
          fold_input=fold_input,
          buckets=buckets,
          ccd=ccd,
          verbose=True,
          ref_max_modified_date=ref_max_modified_date,
          conformer_max_iterations=conformer_max_iterations,
          masking_config=masking_config,
      )
      print(
          f'Featurising data for design took'
          f' {time.time() - featurisation_start_time:.2f} seconds.'
      )
      
      # Initialize list to store results for each seed
      all_inference_results = []
      
      # --- Loop over seeds for design ---
      for design_seed, initial_feature_dict in zip(fold_input.rng_seeds, featurised_examples):
        print(f'--- Starting design process for seed {design_seed} ---')
        logging.info(f"Running binder design for seed {design_seed}")
      
        # Run the design for the current seed
        design_start_time = time.time()
        design_results, final_model_result, final_feature_dict, new_fold_input = binder_design.design_binder(
            fold_input=fold_input,
            feature_dict=initial_feature_dict,
            model_runner=model_runner,
            ccd=ccd,
            design_params=design_params,
            rng_seed=design_seed,
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
            use_complete_prediction=True,
            output_dir=output_dir
        )
        current_design_time = time.time() - design_start_time
        print(f'Binder design and final prediction for seed {design_seed} took {current_design_time:.2f} seconds.')
        logging.info(f"Design for seed {design_seed} finished in {current_design_time:.2f} seconds.")
        
        # Extract results using the FINAL feature dict and model result for this seed
        inference_results, embeddings = model_runner.extract_inference_results_and_maybe_embeddings(
            batch=final_feature_dict, # Use the final features corresponding to the designed sequence
            result=final_model_result, # Use the result from the final prediction run
            target_name=new_fold_input.name # Use the name from the new input (might include _designed suffix)
        )
        
        # Append ResultsForSeed for this seed's final prediction
        all_inference_results.append(
            ResultsForSeed(
                seed=design_seed,
                inference_results=inference_results,
                full_fold_input=new_fold_input, # CRITICAL: Use the new fold input with the designed sequence for this seed
                embeddings=embeddings if _SAVE_EMBEDDINGS.value else None
            )
        )
        
        # Write design-specific outputs (like trajectory) for this seed
        print(f'Writing design-specific outputs for seed {design_seed}...')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the design results (trajectory, losses etc.) as JSON, specific to the seed
        import json
        # Use the potentially modified name from new_fold_input
        sanitised_name = new_fold_input.sanitised_name() 
        design_output_path = os.path.join(output_dir, f'{sanitised_name}_seed_{design_seed}_design_results.json')
        with open(design_output_path, 'w') as f:
            # Convert arrays to lists for JSON serialization
            json_safe_results = {}
            for k, v in design_results.items():
                if k == 'trajectory':
                    # Special handling for trajectory to make it JSON-serializable
                    json_safe_trajectory = {}
                    for tk, tv in v.items():
                        if tk == 'losses' or tk == 'sequences': # Keep these as lists of dicts/strings
                            json_safe_trajectory[tk] = tv
                        elif isinstance(tv, (np.ndarray, jnp.ndarray)):
                            json_safe_trajectory[tk] = tv.tolist() # Convert arrays
                        else:
                            json_safe_trajectory[tk] = tv
                    json_safe_results[k] = json_safe_trajectory
                elif isinstance(v, (np.ndarray, jnp.ndarray)):
                     # Convert other arrays if needed, skip large ones
                    if k not in ['best_feature_dict', 'final_seq_logits']: # Example of skipping large/unserializable items
                      try:
                          json_safe_results[k] = v.tolist()
                      except AttributeError: # Handle non-array types that might sneak in
                          json_safe_results[k] = v
                elif k not in ['best_feature_dict', 'final_seq_logits']:
                    # Keep other items as they are (protocol, best_loss, etc.)
                    json_safe_results[k] = v
            
            # Add the final designed sequence explicitly for easy access
            # Need to find binder chain indices and sequences dynamically
            binder_chain_ids = design_params.get('binder_chains', [])
            
            # Store all designed sequences with chain IDs as keys
            designed_sequences = {}
            
            # Find and store sequences for all binder chains
            for binder_chain_id in binder_chain_ids:
                binder_chain_index = -1
                for idx, chain in enumerate(new_fold_input.chains):
                    if chain.id == binder_chain_id:
                        binder_chain_index = idx
                        break
                
                if binder_chain_index != -1:
                    designed_sequence = new_fold_input.chains[binder_chain_index].sequence
                    designed_sequences[binder_chain_id] = designed_sequence
                else:
                    logging.warning(f"Could not find binder chain ID '{binder_chain_id}' in new_fold_input for seed {design_seed}")
            
            # Add all designed sequences to the results
            if designed_sequences:
                json_safe_results['final_designed_sequences'] = designed_sequences
                # For backward compatibility, also include the first chain sequence under the old key
                if binder_chain_ids:
                    first_chain_id = binder_chain_ids[0]
                    if first_chain_id in designed_sequences:
                        json_safe_results['final_designed_sequence'] = designed_sequences[first_chain_id]
            else:
                logging.warning(f"No binder chain sequences were found for seed {design_seed}")


            json.dump(json_safe_results, f, indent=2, default=lambda x: '<not serializable>')
        print(f'Saved design results summary for seed {design_seed} to {design_output_path}')
      
      # --- End loop over seeds ---

      # Standard output writing (will now use the potentially multiple ResultsForSeed collected)
      print(f'Writing final predicted structure(s) for all seeds...')
      # The write_outputs function should handle the list `all_inference_results`
      output = all_inference_results # Pass the list of results
      
    else:
      # --- Standard Prediction Logic ---
      print('Running standard prediction protocol...')
      logging.info("Running standard prediction...")
      
      # Get CCD
      ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
      
      # Featurize the input
      featurisation_start_time = time.time()
      featurised_examples = featurisation.featurise_input(
          fold_input=fold_input,
          buckets=buckets,
          ccd=ccd,
          verbose=True,
          ref_max_modified_date=ref_max_modified_date,
          conformer_max_iterations=conformer_max_iterations,
          masking_config=masking_config,
      )
      print(
          f'Featurising data took {time.time() - featurisation_start_time:.2f}'
          ' seconds.'
      )

      all_inference_results = []
      inference_start_time = time.time()
      # --- Loop over seeds for prediction ---
      for rng_seed, featurised_example in zip(
          fold_input.rng_seeds, featurised_examples
      ):
        print(f'Running prediction with seed {rng_seed}...')
        logging.info(f"Running prediction with seed {rng_seed}...")
        rng_key = jax.random.PRNGKey(rng_seed)
        model_result = model_runner.run_inference(featurised_example, rng_key)

        inference_results, embeddings = (
            model_runner.extract_inference_results_and_maybe_embeddings(
                batch=featurised_example,
                result=model_result,
                target_name=fold_input.name,
            )
        )
        all_inference_results.append(
            ResultsForSeed(
                seed=rng_seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
                embeddings=embeddings if _SAVE_EMBEDDINGS.value else None
            )
        )
      # --- End loop over seeds ---
      output = all_inference_results
      print(
          f'Model inference took'
          f' {time.time() - inference_start_time:.2f} seconds.'
      )

  # Write outputs (either list of ResultsForSeed or the modified fold_input)
  if isinstance(output, Sequence) and all(
      isinstance(item, ResultsForSeed) for item in output
  ):
    write_outputs(output, output_dir, fold_input.name)
  elif isinstance(output, folding_input.Input):
    # This path is only taken if model_runner was None
    pass
  else:
      logging.warning(f"Unexpected type for 'output' variable: {type(output)}. Skipping output writing.")


  print(f'Fold job {fold_input.name} done, output written to {output_dir}')
  logging.info(f"Fold job {fold_input.name} finished. Output at: {output_dir}")
  return output


def main(_):
  # --- Setup JAX Cache ---
  if _JAX_COMPILATION_CACHE_DIR.value is not None:
    logging.info(f"Setting JAX cache directory: {_JAX_COMPILATION_CACHE_DIR.value}")
    jax.config.update(
        'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
    )
  # -----------------------

  # --- Input Path Validation ---
  if _JSON_PATH.value is None == _INPUT_DIR.value is None:
    raise app.UsageError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )
  # ---------------------------

  # --- Stage Run Validation ---
  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise app.UsageError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )
  # --------------------------

  # --- Load Inputs ---
  if _INPUT_DIR.value is not None:
    logging.info(f"Loading inputs from directory: {_INPUT_DIR.value}")
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(_INPUT_DIR.value)
    )
  elif _JSON_PATH.value is not None:
    logging.info(f"Loading input from file: {_JSON_PATH.value}")
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(_JSON_PATH.value)
    )
  else:
    # This case should be caught by the earlier validation
    raise AssertionError('Input path logic error.')
  logging.info(f"Loaded {fold_inputs} fold input(s).")
  # -----------------

  # --- Create Output Directory ---
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    logging.info(f"Ensured output directory exists: {_OUTPUT_DIR.value}")
  except OSError as e:
    logging.error(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise
  # -----------------------------

  # --- GPU/Device Validation (only if running inference) ---
  if _RUN_INFERENCE.value:
    logging.info("Checking GPU compatibility for inference...")
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
      target_device = gpu_devices[_GPU_DEVICE.value]
      logging.info(f"Using GPU device {_GPU_DEVICE.value}: {target_device}")
      compute_capability = float(target_device.compute_capability)
      logging.info(f"GPU Compute Capability: {compute_capability}")

      if compute_capability < 6.0:
        raise ValueError(
            'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
            ' https://developer.nvidia.com/cuda-gpus).'
        )
      elif 7.0 <= compute_capability < 8.0:
        xla_flags = os.environ.get('XLA_FLAGS', '')
        required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
        if required_flag not in xla_flags:
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
              f' include "{required_flag}". Current XLA_FLAGS: "{xla_flags}"'
          )
        if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the'
              ' --flash_attention_implementation must be set to "xla".'
          )
      # Add checks for Triton/cuDNN compatibility if needed (e.g., Ampere+)
      elif compute_capability < 8.0 and _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
          logging.warning(f"Flash attention implementation '{_FLASH_ATTENTION_IMPLEMENTATION.value}' "
                          f"may not be optimal or compatible with compute capability {compute_capability}. "
                          "Consider using 'xla' or upgrading hardware.")
    else:
         logging.warning("No GPU devices found by JAX. Inference will run on CPU, which is extremely slow.")
  # ------------------------------------------------------

  # --- Print Notice ---
  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n' + '\n'.join(notice) + '\n')
  # ------------------

  # --- Prepare Data Pipeline Config ---
  max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
  if _RUN_DATA_PIPELINE.value:
    logging.info("Preparing data pipeline configuration...")
    expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
    try:
        data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
            nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
            hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
            hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
            hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
            small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
            mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
            uniprot_cluster_annot_database_path=expand_path(
                _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
            ),
            uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
            ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
            rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
            rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
            pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
            seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
            jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
            nhmmer_n_cpu=_NHMMER_N_CPU.value,
            max_template_date=max_template_date,
        )
        logging.info("Data pipeline configuration prepared successfully.")
    except Exception as e:
        logging.error(f"Failed to prepare data pipeline config: {e}", exc_info=True)
        raise
  else:
    data_pipeline_config = None
    logging.info("Data pipeline execution is disabled.")
  # ----------------------------------

  # --- Prepare Model Runner ---
  if _RUN_INFERENCE.value:
    logging.info("Preparing model runner...")
    devices = jax.local_devices(backend='gpu')
    if not devices: # Fallback to CPU if no GPU found/specified
        logging.warning("No GPU found, using CPU for inference (will be slow).")
        devices = jax.local_devices(backend='cpu')
        if not devices:
            raise RuntimeError("No JAX devices (GPU or CPU) available for inference.")

    target_device_for_runner = devices[_GPU_DEVICE.value % len(devices)] # Use modulo for safety
    logging.info(f"Using device {target_device_for_runner} for model runner.")

    print('Building model from scratch...') # Keep user informed
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
            ),
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
        ),
        device=target_device_for_runner,
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...') # Keep user informed
    try:
        _ = model_runner.model_params
        logging.info("Model parameters loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model parameters from {MODEL_DIR.value}: {e}", exc_info=True)
        raise
  else:
    model_runner = None
    logging.info("Model inference is disabled.")
  # --------------------------

  # --- Prepare Masking Config ---
  # This is prepared once and passed to each job
  masking_config = {
      'positions_str': FLAGS.mask_positions,
      'mask_msa': FLAGS.mask_msa,
      'mask_deletion_matrix': FLAGS.mask_deletion_matrix,
      'mask_token': FLAGS.mask_token
  }
  logging.info(f"Masking configuration prepared: {masking_config}")
  # ----------------------------

  # --- Prepare Design Parameters ---
  design_params = None
  if FLAGS.protocol in [DesignProtocol.BINDER_GRADIENT, DesignProtocol.BINDER_BOLTZ]:
    if not FLAGS.target_chains:
      raise app.UsageError("--target_chains required for binder protocols.")
    if not FLAGS.binder_chains:
      raise app.UsageError("--binder_chains required for binder protocols.")

    design_params = {
        "protocol": FLAGS.protocol.value,
        "target_chains": FLAGS.target_chains,
        "binder_chains": FLAGS.binder_chains,
        "lr": FLAGS.design_learning_rate,
        "weights": {"seq_entropy": FLAGS.design_seq_entropy_weight}
    }

    if FLAGS.protocol == DesignProtocol.BINDER_GRADIENT:
        logging.info(f"Running Gradient Binder Design: Target={FLAGS.target_chains}, Binder={FLAGS.binder_chains}")
        design_params["steps"] = FLAGS.design_steps
        design_params["weights"].update({
            "plddt": FLAGS.gradient_plddt_weight,
            "pae_inter": FLAGS.gradient_pae_inter_weight,
            "contact": FLAGS.gradient_contact_weight,
        })
    elif FLAGS.protocol == DesignProtocol.BINDER_BOLTZ:
        logging.info(f"Running BoltzDesign1 Binder Protocol: Target={FLAGS.target_chains}, Binder={FLAGS.binder_chains}")
        design_params["stages"] = [
            FLAGS.boltz_stage1_steps,
            FLAGS.boltz_stage2_steps,
            FLAGS.boltz_stage3_steps,
            FLAGS.boltz_stage4_steps
        ]
        design_params["weights"].update({
            "contact_intra": FLAGS.boltz_contact_intra_weight,
            "contact_inter": FLAGS.boltz_contact_inter_weight,
            "confidence": FLAGS.boltz_confidence_weight,
        })
        # Add clear_memory_interval to enable memory optimization
        design_params["clear_memory_interval"] = FLAGS.clear_memory_interval
  else:
    logging.info("Running standard folding protocol.")
  # ---------------------------

  # --- Setup Logging ---
  try:
    from alphafold3.design import logging as af_logging
    af_logging.configure_logging(
        log_level=_LOG_LEVEL.value,
        log_format=_LOG_FORMAT.value,
        log_file=_LOG_FILE.value,
        config_file=_LOG_CONFIG_FILE.value,
    )
    logging.info("Custom logging configured successfully.")
  except ImportError as e:
    # Fallback to basic logging if our module isn't available
    logging.warning(f"Could not import design logging module: {e}. Using default logging.")
    logging.basicConfig(
        level=getattr(logging, _LOG_LEVEL.value.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
  # ----------------------

  # --- Process Each Input ---
  num_fold_inputs_processed = 0
  total_start_time = time.time()
  for fold_input_item in fold_inputs:
    job_start_time = time.time()
    if _NUM_SEEDS.value is not None:
      logging.info(f'Expanding fold job {fold_input_item.name} to {_NUM_SEEDS.value} seeds')
      fold_input_item = fold_input_item.with_multiple_seeds(_NUM_SEEDS.value)

    # Define output directory for this specific job
    job_output_dir = os.path.join(_OUTPUT_DIR.value, fold_input_item.sanitised_name())

    # --- Override binder length if flag is set ---
    if FLAGS.binder_length is not None:
        if not FLAGS.binder_chains:
            raise ValueError("--binder_length requires --binder_chains to be set.")
        logging.info(f"Overriding binder length to {FLAGS.binder_length} for chains {FLAGS.binder_chains}")
        new_chains = []
        binder_chain_ids_to_modify = set(FLAGS.binder_chains)
        for chain in fold_input_item.chains:
            if chain.id in binder_chain_ids_to_modify and isinstance(chain, folding_input.ProteinChain):
                logging.info(f"Modifying length of binder chain {chain.id} to {FLAGS.binder_length}")
                new_sequence = 'A' * FLAGS.binder_length
                # Create a new chain with the overridden sequence and explicitly empty MSA/templates
                modified_chain = folding_input.ProteinChain(
                    id=chain.id,
                    sequence=new_sequence,
                    ptms=[], # Reset PTMs for new sequence
                    unpaired_msa="", # Explicitly empty to skip pipeline
                    paired_msa="",   # Explicitly empty to skip pipeline
                    templates=[]     # Explicitly empty to skip pipeline
                )
                new_chains.append(modified_chain)
            else:
                new_chains.append(chain)
        # Replace chains in the input object (dataclasses are immutable, need replace)
        fold_input_item = dataclasses.replace(fold_input_item, chains=tuple(new_chains))
    # --------------------------------------------

    # --- Override seed if flag is set ---
    if FLAGS.seed is not None:
        logging.info(f"Overriding random seeds with single seed: {FLAGS.seed}")
        fold_input_item = dataclasses.replace(fold_input_item, rng_seeds=(FLAGS.seed,))
    # ----------------------------------

    process_fold_input(
        fold_input=fold_input_item,
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=job_output_dir,
        # --- Pass mutation/masking args ---
        mutations_str=FLAGS.mutations,
        masking_config=masking_config,
        # --- Pass design params ---
        design_params=design_params,
        # ----------------------------------
        buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
        ref_max_modified_date=max_template_date,
        conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
        force_output_dir=_FORCE_OUTPUT_DIR.value,
    )
    num_fold_inputs_processed += 1
    logging.info(f"Finished processing job {fold_input_item.name} in {time.time() - job_start_time:.2f} seconds.")

  total_time = time.time() - total_start_time
  print(f'\nDone running {num_fold_inputs_processed} fold job(s) in {total_time:.2f} seconds.')
  logging.info(f"Finished processing all {num_fold_inputs_processed} jobs in {total_time:.2f} seconds.")
  # ------------------------


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)

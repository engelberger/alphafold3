import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock

# Assume modules are importable from src path
from alphafold3.design import binder_design
from alphafold3.design import binder_utils
from alphafold3.common import folding_input
from alphafold3.design.config import DesignConfig

@pytest.fixture
def dummy_fold_input():
    """Creates a minimal dummy folding_input.Input."""
    # Define chains (adjust lengths as needed)
    chain_a = folding_input.ProteinChain(id='A', sequence='A'*20)
    chain_b = folding_input.ProteinChain(id='B', sequence='C'*15) # Binder
    
    # Create the input object
    return folding_input.Input(
        name="dummy_design_input",
        chains=[chain_a, chain_b],
        rng_seeds=[42], # Single seed for simplicity
    )

@pytest.fixture
def dummy_feature_dict_design():
    """Provides a basic feature dictionary suitable for design setup."""
    num_res_total = 35 # 20 (A) + 15 (B)
    num_res_padded = 64 # Example padding
    msa_depth = 10
    
    # Map asym_id: 0 for chain A, 1 for chain B, 2 for padding
    asym_id = np.concatenate([
        np.zeros(20, dtype=int), 
        np.ones(15, dtype=int), 
        np.ones(num_res_padded - num_res_total, dtype=int) * 2 
    ])
    
    seq_mask = np.concatenate([
        np.ones(num_res_total, dtype=bool), 
        np.zeros(num_res_padded - num_res_total, dtype=bool)
    ])
    
    aatype = np.zeros((num_res_padded,), dtype=int) # Placeholder
    msa = np.zeros((msa_depth, num_res_padded), dtype=int) # Placeholder
    msa_mask = np.ones((msa_depth, num_res_padded), dtype=bool) # Placeholder
    
    return {
        'asym_id': jnp.array(asym_id),
        'seq_mask': jnp.array(seq_mask),
        'aatype': jnp.array(aatype), 
        'msa': jnp.array(msa),
        'msa_mask': jnp.array(msa_mask),
        # Add other minimal keys if setup_binder_features requires them
    }


@pytest.fixture
def mock_model_runner():
    """Creates a mock ModelRunner that returns a fixed result."""
    mock = MagicMock()
    # Define the dummy result structure expected by loss functions
    dummy_result = {
        'predicted_lddt': jnp.ones((1, 64, 14)) * 0.8, # Match padded size
        'full_pae': jnp.ones((1, 64, 64)) * 10.0,
        'distogram': {
            'logits': jnp.zeros((64, 64, 64)),
            'bin_edges': jnp.linspace(2.0, 20.0, 63),
        },
        # Add other keys if loss functions require them
    }
    mock.run_inference.return_value = dummy_result
    return mock


@pytest.mark.baseline
def test_binder_designer_gradient_smoke(dummy_fold_input, dummy_feature_dict_design, mock_model_runner):
    """A smoke test to ensure the gradient design pipeline runs without crashing."""
    
    design_config = DesignConfig(
        protocol_name="binder_gradient",
        target_chains=["A"], # From dummy_fold_input
        binder_chains=["B"], # From dummy_fold_input
        learning_rate=0.1,
        steps=2, # Keep low for speed
        weights={
            "plddt": 0.1,
            "pae_inter": 0.1,
            "contact_intra": 0.0, # Disable complex losses for simplicity
            "contact_inter": 0.0,
            "seq_entropy": 0.01
        },
            clear_memory_interval=0 # Disable memory clearing
    )
    
    # Mock CCD (not strictly needed if features are provided, but good practice)
    mock_ccd = MagicMock()

    designer = binder_design.BinderDesigner(
        model_runner=mock_model_runner,
        ccd=mock_ccd,
        design_config=design_config
    )

    rng_key = jax.random.PRNGKey(0)

    # Run the design process
    # Note: This uses the internal method directly for baseline testing
    # The public API `design_binder` might be tested later
    design_results, final_feature_dict = designer._design_binder_gradient(
        fold_input=dummy_fold_input,
        feature_dict=dummy_feature_dict_design, 
        target_indices=jnp.arange(0, 20), # Manually set based on dummy_fold_input
        binder_indices=jnp.arange(20, 35), # Manually set based on dummy_fold_input
        rng_key=rng_key,
    )

    # Basic assertions to ensure the process ran
    assert isinstance(design_results, dict)
    assert design_results["protocol"] == "binder_gradient"
    assert "best_loss" in design_results
    assert "trajectory" in design_results
    assert len(design_results["trajectory"]["loss"]) == design_config.steps
    assert len(design_results["trajectory"]["sequences"]) > 0 # Should generate sequences
    assert isinstance(final_feature_dict, dict)

    # Check that the mock runner was called
    assert mock_model_runner.run_inference.call_count == design_config.steps

# Placeholder for Boltz baseline test (more complex to mock)
# @pytest.mark.baseline
# def test_binder_designer_boltz_smoke(...) 
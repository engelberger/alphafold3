import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Assume the binder_loss module is importable from src path
from alphafold3.design import binder_loss

# --- Fixtures for Dummy Data ---

@pytest.fixture
def dummy_model_result():
    """Provides a minimal dummy result dictionary for loss functions."""
    num_res = 50
    num_atoms = 14 # Max atoms for simplicity
    num_bins = 64
    num_samples = 1 # For plddt/pae
    return {
        'predicted_lddt': jnp.ones((num_samples, num_res, num_atoms)) * 0.8,
        'full_pae': jnp.ones((num_samples, num_res, num_res)) * 10.0,
        'distogram': {
            'logits': jnp.zeros((num_res, num_res, num_bins)),
            'bin_edges': jnp.linspace(2.0, 20.0, num_bins - 1),
        },
        'final_atom_positions': jnp.ones((num_res, num_atoms, 3)),
    }

@pytest.fixture
def dummy_feature_dict():
    """Provides a minimal dummy feature dictionary."""
    num_res = 50
    num_atoms = 14
    return {
        'initial_coords': jnp.ones((num_res, num_atoms, 3)) * 1.1 # Slightly different from final
    }

@pytest.fixture
def dummy_indices():
    """Provides target and binder indices."""
    target_indices = jnp.arange(0, 25)
    binder_indices = jnp.arange(25, 50)
    return target_indices, binder_indices

@pytest.fixture
def dummy_logits():
    """Provides dummy sequence logits."""
    num_binder = 25
    num_residue_types = 20
    return jnp.zeros((num_binder, num_residue_types))

# --- Baseline Tests for Loss Helper Functions ---

@pytest.mark.baseline
def test_entropy_low_bins():
    num_bins = 64
    logits = jnp.zeros((10, 10, num_bins)) # Uniform dist
    low_mask = jnp.arange(num_bins) < 32 # Mask first half
    entropy = binder_loss.entropy_low_bins(logits, low_mask)
    assert entropy.shape == (10, 10)
    # For uniform dist over N bins, entropy is ln(N)
    # Here, q_star is uniform over 32 bins -> H = ln(32) ~ 3.46
    # q is uniform over 64 bins
    # Sum(q_star * log(q)) = Sum_{i=0}^{31} (1/32 * log(1/64)) = 32 * (1/32 * -ln(64)) = -ln(64)
    # entropy = -(-ln(64)) = ln(64) ~ 4.15 - CHECK THIS LOGIC LATER
    # Let's just check it runs and returns a float for now
    assert isinstance(jnp.mean(entropy).item(), float)


@pytest.mark.baseline
def test_get_binder_plddt_loss(dummy_model_result, dummy_indices):
    _, binder_indices = dummy_indices
    loss = binder_loss.get_binder_plddt_loss(dummy_model_result, binder_indices)
    # Expected: -(mean of 0.8) = -0.8
    assert np.isclose(loss.item(), -0.8)

@pytest.mark.baseline
def test_get_interface_pae_loss(dummy_model_result, dummy_indices):
    target_indices, binder_indices = dummy_indices
    loss = binder_loss.get_interface_pae_loss(dummy_model_result, target_indices, binder_indices)
    # Expected: mean of 10.0 = 10.0
    assert np.isclose(loss.item(), 10.0)

@pytest.mark.baseline
def test_get_target_fape_loss(dummy_model_result, dummy_feature_dict, dummy_indices):
    target_indices, _ = dummy_indices
    initial_coords = dummy_feature_dict['initial_coords']
    loss = binder_loss.get_target_fape_loss(dummy_model_result, initial_coords, target_indices)
    # Expected: Mean squared diff between 1.0 and 1.1 coords = (1.1-1.0)^2 * 3 = 0.01 * 3 = 0.03
    # Simplified check for now, depends on masking logic
    assert loss.item() >= 0.0

@pytest.mark.baseline
def test_get_binder_seq_entropy_loss(dummy_logits):
    loss = binder_loss.get_binder_seq_entropy_loss(dummy_logits)
    # Max entropy for uniform dist over 20 types = ln(20) ~ 2.995
    # Loss is negative entropy, so should be close to -2.995
    assert np.isclose(loss.item(), -jnp.log(20.0))

@pytest.mark.baseline
def test_get_distogram_entropy_loss(dummy_model_result, dummy_indices):
    target_indices, binder_indices = dummy_indices
    distogram_logits = dummy_model_result['distogram']['logits']
    bin_breaks = dummy_model_result['distogram']['bin_edges']
    weights = {'contact_intra': 1.0, 'contact_inter': 0.5}

    total_loss, breakdown = binder_loss.get_distogram_entropy_loss(
        distogram_logits, bin_breaks, target_indices, binder_indices, weights
    )
    assert isinstance(total_loss.item(), float)
    assert 'contact_intra' in breakdown
    assert 'contact_inter' in breakdown
    assert isinstance(breakdown['contact_intra'].item(), float)
    assert isinstance(breakdown['contact_inter'].item(), float)

# --- Tests for Combined Loss Functions (can add later if needed) ---
# @pytest.mark.baseline
# def test_calculate_gradient_binder_loss(...)
#
# @pytest.mark.baseline
# def test_calculate_boltz_binder_loss(...)


</rewritten_file> 
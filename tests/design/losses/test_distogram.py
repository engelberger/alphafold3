import pytest
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.design.losses import distogram
from alphafold3.design.losses.common import entropy_low_bins

@pytest.fixture
def dummy_distogram_data():
    num_res = 15
    num_bins = 64
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (num_res, num_res, num_bins))
    bin_breaks = jnp.linspace(2.0, 20.0, num_bins - 1)
    target_indices = jnp.arange(7)
    binder_indices = jnp.arange(7, 15)
    weights = {'contact_intra': 1.0, 'contact_inter': 0.8}
    return logits, bin_breaks, target_indices, binder_indices, weights

@pytest.mark.parametrize("num_bins, cutoff_fraction", [
    (64, 0.5), # Mask half
    (64, 0.25), # Mask quarter
    (64, 1.0), # Mask all (entropy should be high)
    (64, 0.0), # Mask none (entropy should be ~log(N))
])
def test_entropy_low_bins_param(num_bins, cutoff_fraction):
    """Test entropy calculation with varying bin masks."""
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (5, 5, num_bins)) # Smaller size for speed
    
    # Create mask
    cutoff_bin = int(num_bins * cutoff_fraction)
    low_mask = jnp.arange(num_bins) < cutoff_bin
    
    @jax.jit
    def compiled_entropy_fn(lgts, mask):
        return entropy_low_bins(lgts, mask)
        
    entropy = compiled_entropy_fn(logits, low_mask)
    assert entropy.shape == (5, 5)
    assert jnp.all(jnp.isfinite(entropy)) # Ensure no NaNs/Infs


def test_get_distogram_entropy_loss_jit(dummy_distogram_data):
    """Test get_distogram_entropy_loss under JIT compilation."""
    logits, bin_breaks, target_indices, binder_indices, weights = dummy_distogram_data

    @jax.jit
    def compiled_loss_fn(lgts, breaks, tgt_idx, bnd_idx, w):
        return distogram.get_distogram_entropy_loss(
            lgts, breaks, tgt_idx, bnd_idx, w
        )

    total_loss, breakdown = compiled_loss_fn(
        logits, bin_breaks, target_indices, binder_indices, weights
    )

    assert isinstance(total_loss.item(), float)
    assert total_loss.item() >= 0 # Entropy loss should be non-negative
    assert 'contact_intra' in breakdown
    assert 'contact_inter' in breakdown
    assert isinstance(breakdown['contact_intra'].item(), float)
    assert isinstance(breakdown['contact_inter'].item(), float)
    assert breakdown['contact_intra'].item() >= 0
    assert breakdown['contact_inter'].item() >= 0

# Test edge case with zero weights
def test_get_distogram_entropy_loss_zero_weights(dummy_distogram_data):
    logits, bin_breaks, target_indices, binder_indices, _ = dummy_distogram_data
    zero_weights = {'contact_intra': 0.0, 'contact_inter': 0.0}

    @jax.jit
    def compiled_loss_fn(lgts, breaks, tgt_idx, bnd_idx, w):
        return distogram.get_distogram_entropy_loss(
            lgts, breaks, tgt_idx, bnd_idx, w
        )
        
    total_loss, breakdown = compiled_loss_fn(
        logits, bin_breaks, target_indices, binder_indices, zero_weights
    )
    assert np.isclose(total_loss.item(), 0.0)
    assert np.isclose(breakdown['contact_intra'].item(), 0.0)
    assert np.isclose(breakdown['contact_inter'].item(), 0.0) 
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.design.losses import pae

# Minimal fixture for PAE tests
@pytest.fixture
def dummy_pae_result():
    num_res = 20
    # Simulate both batched and non-batched PAE
    key = jax.random.PRNGKey(0)
    pae_matrix = jax.random.uniform(key, (num_res, num_res), minval=1.0, maxval=30.0)
    pae_matrix_batched = jax.random.uniform(key, (2, num_res, num_res), minval=1.0, maxval=30.0)
    confidence_matrix = jax.random.uniform(key+1, (num_res, num_res), minval=0.5, maxval=1.0)
    
    return {
        'full_pae': pae_matrix, 
        'batched_pae': pae_matrix_batched, 
        'predicted_aligned_error_confidence': confidence_matrix,
        'predicted_aligned_error': pae_matrix # Alias for calculate_pae_loss
    }

@pytest.fixture
def pae_indices():
    target_indices = jnp.arange(10)
    binder_indices = jnp.arange(10, 20)
    return target_indices, binder_indices

# Test get_pae_matrix extraction
@pytest.mark.parametrize("key, expect_found", [
    ('full_pae', True),
    ('pae', True), # Assuming alias exists in result
    ('missing_key', False),
])
def test_get_pae_matrix(key, expect_found):
    num_res = 5
    result = {}
    if key != 'missing_key':
        result[key] = jnp.ones((num_res, num_res))
        
    matrix = pae.get_pae_matrix(result)
    if expect_found:
        assert matrix is not None
        assert matrix.shape == (num_res, num_res)
    else:
        assert matrix is None

# Test process_pae_dimensions
@pytest.mark.parametrize("input_shape", [
    (20, 20), # Already 2D
    (1, 20, 20), # Single batch dim
    (2, 20, 20), # Multi batch dim
])
def test_process_pae_dimensions(input_shape):
    pae_matrix = jnp.ones(input_shape)
    processed = pae.process_pae_dimensions(pae_matrix)
    assert processed.shape == (20, 20)
    # Check if mean was taken for batch dim
    if len(input_shape) > 2:
        assert jnp.allclose(processed, jnp.mean(pae_matrix, axis=0))
    else:
        assert jnp.allclose(processed, pae_matrix)


def test_get_interface_pae_loss_jit(dummy_pae_result, pae_indices):
    """Test get_interface_pae_loss under JIT compilation."""
    target_indices, binder_indices = pae_indices
    
    @jax.jit
    def compiled_loss_fn(result, tgt_idx, bnd_idx):
        return pae.get_interface_pae_loss(result, tgt_idx, bnd_idx)
        
    loss = compiled_loss_fn(dummy_pae_result, target_indices, binder_indices)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0 # PAE loss should be positive
    
    # Test with batched input
    batched_result = {'full_pae': dummy_pae_result['batched_pae']}
    loss_batched = compiled_loss_fn(batched_result, target_indices, binder_indices)
    assert isinstance(loss_batched.item(), float)
    assert loss_batched.item() >= 0


def test_get_masked_pae_loss_jit(dummy_pae_result, pae_indices):
    """Test get_masked_pae_loss under JIT compilation."""
    target_indices, binder_indices = pae_indices
    
    @jax.jit
    def compiled_loss_fn(result, rows, cols):
        return pae.get_masked_pae_loss(result, rows, cols)
        
    # Interface loss
    loss_interface = compiled_loss_fn(dummy_pae_result, target_indices, binder_indices)
    assert isinstance(loss_interface.item(), float)
    assert loss_interface.item() >= 0
    
    # Intra-target loss
    loss_intra_target = compiled_loss_fn(dummy_pae_result, target_indices, target_indices)
    assert isinstance(loss_intra_target.item(), float)
    assert loss_intra_target.item() >= 0


def test_calculate_pae_loss_jit(dummy_pae_result, pae_indices):
    """Test calculate_pae_loss under JIT compilation."""
    target_indices, binder_indices = pae_indices
    
    # JIT the function, marking interface_only as static
    jitted_loss_fn = jax.jit(
        pae.calculate_pae_loss,
        static_argnames=('interface_only',)
    )
        
    # Interface only
    loss_interface = jitted_loss_fn(dummy_pae_result, binder_indices, target_indices, interface_only=True)
    assert isinstance(loss_interface.item(), float)
    assert loss_interface.item() >= 0
    
    # Full matrix
    loss_full = jitted_loss_fn(dummy_pae_result, binder_indices, target_indices, interface_only=False)
    assert isinstance(loss_full.item(), float)
    assert loss_full.item() >= 0


def test_calculate_pae_confidence_loss_jit(dummy_pae_result, pae_indices):
    """Test calculate_pae_confidence_loss under JIT."""
    target_indices, binder_indices = pae_indices
    
    @jax.jit
    def compiled_loss_fn(result, b_idx, t_idx, threshold):
        return pae.calculate_pae_confidence_loss(result, b_idx, t_idx, threshold)
        
    loss = compiled_loss_fn(dummy_pae_result, binder_indices, target_indices, 0.75)
    assert isinstance(loss.item(), float) >= 0
    
    # Test with higher threshold (likely higher loss or NaN if no pairs qualify)
    loss_high_thresh = compiled_loss_fn(dummy_pae_result, binder_indices, target_indices, 0.95)
    assert isinstance(loss_high_thresh.item(), float)


def test_calculate_max_interface_pae_loss_jit(dummy_pae_result, pae_indices):
    """Test calculate_max_interface_pae_loss under JIT."""
    target_indices, binder_indices = pae_indices
    
    @jax.jit
    def compiled_loss_fn(result, b_idx, t_idx, percentile_static):
        return pae.calculate_max_interface_pae_loss(result, b_idx, t_idx, percentile=percentile_static)
        
    # 90th percentile
    static_loss_fn_90 = jax.jit(compiled_loss_fn, static_argnames=('percentile_static',))
    loss_90 = static_loss_fn_90(dummy_pae_result, binder_indices, target_indices, 90.0)
    assert isinstance(loss_90.item(), float) >= 0
    
    # Max (100th percentile)
    static_loss_fn_100 = jax.jit(compiled_loss_fn, static_argnames=('percentile_static',))
    loss_100 = static_loss_fn_100(dummy_pae_result, binder_indices, target_indices, 100.0)
    assert isinstance(loss_100.item(), float) >= 0
    # Max should be >= 90th percentile
    assert loss_100.item() >= loss_90.item() 
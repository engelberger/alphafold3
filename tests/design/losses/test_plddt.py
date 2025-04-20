import pytest
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.design.losses import plddt

# Reusing dummy data fixtures from baseline tests might be useful here
# For now, creating minimal specific fixtures

@pytest.fixture
def dummy_plddt_result():
    num_samples = 2
    num_res = 10
    num_atoms = 5
    # Create plddt scores (0-100 range)
    key = jax.random.PRNGKey(0)
    plddt_scores = jax.random.uniform(key, (num_samples, num_res, num_atoms), minval=50., maxval=95.)
    return {'predicted_lddt': plddt_scores}


@pytest.mark.parametrize("binder_indices", [
    jnp.arange(5), # First 5 residues
    jnp.array([0, 2, 4, 6, 8]), # Non-contiguous
    jnp.arange(10) # All residues
])
def test_get_binder_plddt_loss_jit(dummy_plddt_result, binder_indices):
    """Test get_binder_plddt_loss under JIT compilation."""
    
    @jax.jit
    def compiled_loss_fn(result, indices):
        return plddt.get_binder_plddt_loss(result, indices)
    
    loss = compiled_loss_fn(dummy_plddt_result, binder_indices)
    
    assert isinstance(loss.item(), float)
    assert loss.item() <= 0 # Should be negative mean plddt


@pytest.mark.parametrize("mode, threshold, selection_indices, expected_sign", [
    ("maximize", 70.0, None, -1), # Maximize all -> negative loss
    ("maximize", 70.0, jnp.arange(5), -1), # Maximize subset -> negative loss
    ("threshold", 80.0, None, 1), # Threshold -> positive loss (MSE)
    ("threshold", 80.0, jnp.arange(5), 1), # Threshold subset -> positive loss (MSE)
])
def test_calculate_plddt_loss_jit(dummy_plddt_result, mode, threshold, selection_indices, expected_sign):
    """Test calculate_plddt_loss under JIT compilation."""
    
    # We need to JIT the function we actually call, marking the mode arg as static
    # Note: The function itself uses Python 'if' on mode, so it MUST be static.
    jitted_loss_fn = jax.jit(
        plddt.calculate_plddt_loss, 
        static_argnames=('mode',)
    )

    # Pass the mode directly to the jitted function
    loss = jitted_loss_fn(dummy_plddt_result, selection_indices, threshold, mode=mode)

    assert isinstance(loss.item(), float)
    if expected_sign < 0:
        assert loss.item() < 0
    else:
        assert loss.item() >= 0 # MSE loss >= 0


def test_calculate_plddt_confidence_weighted_loss_jit(dummy_plddt_result):
    """Test calculate_plddt_confidence_weighted_loss under JIT."""
    num_res = dummy_plddt_result['predicted_lddt'].shape[1]
    key = jax.random.PRNGKey(1)
    confidence_weights = jax.random.uniform(key, (num_res,))
    selection_indices = jnp.arange(num_res // 2)

    @jax.jit
    def compiled_loss_fn(result, weights, indices):
        return plddt.calculate_plddt_confidence_weighted_loss(result, weights, indices)

    loss = compiled_loss_fn(dummy_plddt_result, confidence_weights, selection_indices)

    assert isinstance(loss.item(), float)
    assert loss.item() <= 0 # Should be negative weighted mean 
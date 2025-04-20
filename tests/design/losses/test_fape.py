import pytest
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.design.losses import fape

@pytest.fixture
def dummy_fape_data():
    num_res = 10
    num_atoms = 5
    key = jax.random.PRNGKey(0)
    
    result = {
        'final_atom_positions': jax.random.normal(key, (num_res, num_atoms, 3))
    }
    initial_coords = jax.random.normal(key+1, (num_res, num_atoms, 3))
    target_indices = jnp.arange(num_res // 2) # First half
    
    return result, initial_coords, target_indices

def test_get_target_fape_loss_jit(dummy_fape_data):
    """Test get_target_fape_loss under JIT."""
    result, initial_coords, target_indices = dummy_fape_data
    
    @jax.jit
    def compiled_loss_fn(res, init_coords, tgt_idx):
        return fape.get_target_fape_loss(res, init_coords, tgt_idx)
        
    loss = compiled_loss_fn(result, initial_coords, target_indices)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0 # FAPE loss should be non-negative

def test_get_target_fape_loss_missing_key(dummy_fape_data):
    """Test behavior when final_atom_positions is missing."""
    result, initial_coords, target_indices = dummy_fape_data
    # Remove the required key
    del result['final_atom_positions']
    
    loss = fape.get_target_fape_loss(result, initial_coords, target_indices)
    # Should return 0.0 and log a warning
    assert np.isclose(loss.item(), 0.0)

def test_get_target_fape_loss_identical_coords(dummy_fape_data):
    """Test behavior when coordinates are identical (zero loss)."""
    result, initial_coords, target_indices = dummy_fape_data
    # Make coords identical
    result['final_atom_positions'] = initial_coords
    
    @jax.jit
    def compiled_loss_fn(res, init_coords, tgt_idx):
        return fape.get_target_fape_loss(res, init_coords, tgt_idx)
        
    loss = compiled_loss_fn(result, initial_coords, target_indices)
    assert np.isclose(loss.item(), 0.0) 
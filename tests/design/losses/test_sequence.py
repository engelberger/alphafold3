import pytest
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.design.losses import sequence

@pytest.fixture
def dummy_seq_logits():
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (10, 20)) # 10 residues, 20 types


def test_get_binder_seq_entropy_loss_jit(dummy_seq_logits):
    """Test get_binder_seq_entropy_loss under JIT."""
    
    @jax.jit
    def compiled_loss_fn(logits):
        return sequence.get_binder_seq_entropy_loss(logits)
        
    loss = compiled_loss_fn(dummy_seq_logits)
    
    assert isinstance(loss.item(), float)
    # Max entropy is log(20) ~ 2.995, so min loss is -log(20)
    # Should be less than or equal to 0
    assert loss.item() <= 1e-6 # Allow for small positive due to float precision
    assert loss.item() > -3.0 # Ensure it's not excessively negative 
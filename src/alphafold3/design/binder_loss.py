"""Loss functions for AlphaFold 3 design protocols.

This module re-exports loss functions from the `alphafold3.design.losses` package
for easier access and backward compatibility.

New code should ideally import directly from the specific submodules
(e.g., `alphafold3.design.losses.plddt`).
"""

import warnings

# Re-export functions from the new structure
from alphafold3.design.losses import (
    # Common utils (use sparingly)
    entropy_low_bins,
    safe_mean,
    
    # Specific loss components
    get_binder_plddt_loss,
    calculate_plddt_loss,
    calculate_plddt_confidence_weighted_loss,
    get_interface_pae_loss,
    calculate_pae_loss,
    calculate_pae_confidence_loss,
    calculate_max_interface_pae_loss,
    get_distogram_entropy_loss,
    get_binder_seq_entropy_loss,
    get_target_fape_loss,
    
    # Combined protocol losses
    calculate_gradient_binder_loss,
    calculate_boltzdesign_binder_loss, # Note: renamed from calculate_boltz_binder_loss
)

# Emit a warning to encourage using the new structure
warnings.warn(
    "Importing from `alphafold3.design.binder_loss` is deprecated. "
    "Please import directly from `alphafold3.design.losses` submodules.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'entropy_low_bins',
    'safe_mean',
    'get_binder_plddt_loss',
    'calculate_plddt_loss',
    'calculate_plddt_confidence_weighted_loss',
    'get_interface_pae_loss',
    'calculate_pae_loss',
    'calculate_pae_confidence_loss',
    'calculate_max_interface_pae_loss',
    'get_distogram_entropy_loss',
    'get_binder_seq_entropy_loss',
    'get_target_fape_loss',
    'calculate_gradient_binder_loss',
    'calculate_boltzdesign_binder_loss',
]


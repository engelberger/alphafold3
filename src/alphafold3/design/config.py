import dataclasses
from typing import List, Dict, Optional, Sequence

# Define alphabet size for standard amino acids
ALPHABET_SIZE = 20

# Default values primarily derived from flag definitions in run_alphafold.py

@dataclasses.dataclass(frozen=True)
class LossWeightsConfig:
    """Configuration for loss term weights."""
    # Common
    seq_entropy: float = 0.00 # Default from --design_seq_entropy_weight

    # Gradient Protocol Specific (--gradient_*)
    gradient_plddt: float = 0.5
    gradient_pae_inter: float = 0.5
    # Note: --gradient_contact_weight=0.5 likely maps to inter-contact loss
    # The gradient loss function sums weighted intra and inter. We'll map
    # the flag to inter here and assume intra defaults to 0 unless overridden.
    gradient_contact_inter: float = 0.0
    gradient_contact_intra: float = 0.0 # Defaulting to 0 as only one flag exists
    gradient_fape_target: float = 0.0 # Hidden parameter, default 0.0

    # Boltz Protocol Specific (--boltz_*)
    boltz_contact_intra: float = 0.0
    boltz_contact_inter: float = 0.0
    boltz_confidence: float = 1.0 # Combined pLDDT + PAE weight
    boltz_helix: float = 0.0 # Placeholder, default 0.0


@dataclasses.dataclass(frozen=True)
class BoltzDesignConfig:
    """Configuration specific to the BoltzDesign protocol."""
    # Stage lengths default to flag values
    stages: List[int] = dataclasses.field(default_factory=lambda: [50, 50, 50, 50])
    # Hidden parameter default from internal code
    random_init_scale: float = 0.01
    # Placeholder for helicity bias (hidden parameter)
    helicity_bias: float = 0.0
    # Contains weights relevant to Boltz protocol
    weights: LossWeightsConfig = dataclasses.field(default_factory=LossWeightsConfig)

    # STE schedule parameters
    ste_alpha: float = 1.0
    ste_temp_start: float = 1.0
    ste_temp_end: float = 0.01 # Needs low temp for STE stage
    ste_soft_start: float = 0.0
    ste_soft_end: float = 0.0
    ste_hard_start: float = 0.0
    ste_hard_end: float = 1.0 # STE stage is hard


@dataclasses.dataclass(frozen=True)
class GradientDesignConfig:
    """Configuration specific to the Gradient protocol."""
    # Steps default to flag value
    steps: int = 200
    # Contains weights relevant to Gradient protocol
    weights: LossWeightsConfig = dataclasses.field(default_factory=LossWeightsConfig)

    # STE schedule parameters
    ste_alpha: float = 1.0
    ste_temp_start: float = 1.0
    ste_temp_end: float = 0.1 # Ramp down temperature
    ste_soft_start: float = 1.0 # Start soft
    ste_soft_end: float = 0.0
    ste_hard_start: float = 0.0
    ste_hard_end: float = 1.0 # End hard (STE)

    # Optimizer choice
    optimizer_name: str = "adam" # Default to adam, read from flag in run_alphafold.py


@dataclasses.dataclass(frozen=True)
class DesignConfig:
    """Top-level configuration for binder design."""
    # Populated based on --protocol flag
    protocol_name: str  # e.g., "binder_boltz", "binder_gradient"

    # Chains - required, no default here, checked in run_alphafold.py
    target_chains: List[str]
    binder_chains: List[str]

    # Common parameters from flags
    learning_rate: float = 0.1  # Default from --design_learning_rate
    clear_memory_interval: int = 0  # Default from --clear_memory_interval
    optimizer_name: str = "adam"  # Name of optimizer to use for sequence design

    # Logging & tracking parameters
    verbosity: int = 1  # How often to print log lines (every N steps)
    best_metric: str = "loss"  # Metric name to track for saving best state
    traj_max: int = 10  # Maximum length of stored trajectory per metric

    # Protocol-specific configuration
    # One of these will be populated based on protocol_name
    boltz_config: Optional[BoltzDesignConfig] = None
    gradient_config: Optional[GradientDesignConfig] = None

    # Optional: Add fields for parameters currently lacking flags if needed
    # e.g., helicity_bias: float = 0.0 (though currently unused)

    def get_weights(self) -> LossWeightsConfig:
        """Helper to get the relevant weights config."""
        if self.protocol_name == "binder_boltz" and self.boltz_config:
            return self.boltz_config.weights
        elif self.protocol_name == "binder_gradient" and self.gradient_config:
            return self.gradient_config.weights
        else:
            # Should not happen if constructed correctly
            raise ValueError(f"Invalid protocol ({self.protocol_name}) or missing config") 
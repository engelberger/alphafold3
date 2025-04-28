from absl import flags

# Optimizer choice for sequence design
flags.DEFINE_string(
    'design_optimizer', 'adam',
    'Name of the Optax optimizer to use for sequence design (e.g., adam, adamw).'
)

# Gradient protocol STE schedule flags
flags.DEFINE_float('gradient_ste_alpha', 1.0, 'STE alpha for gradient protocol.')
flags.DEFINE_float('gradient_ste_temp_start', 1.0, 'Initial STE temperature for gradient protocol.')
flags.DEFINE_float('gradient_ste_temp_end', 0.1, 'Final STE temperature for gradient protocol.')
flags.DEFINE_float('gradient_ste_soft_start', 1.0, 'Initial soft weight for gradient protocol STE.')
flags.DEFINE_float('gradient_ste_soft_end', 0.0, 'Final soft weight for gradient protocol STE.')
flags.DEFINE_float('gradient_ste_hard_start', 0.0, 'Initial hard weight for gradient protocol STE.')
flags.DEFINE_float('gradient_ste_hard_end', 1.0, 'Final hard weight for gradient protocol STE.')

# Boltz protocol STE schedule flags
flags.DEFINE_float('boltz_ste_alpha', 1.0, 'STE alpha for Boltz protocol.')
flags.DEFINE_float('boltz_ste_temp_start', 1.0, 'Initial STE temperature for Boltz protocol.')
flags.DEFINE_float('boltz_ste_temp_end', 0.01, 'Final STE temperature for Boltz protocol.')
flags.DEFINE_float('boltz_ste_soft_start', 0.0, 'Initial soft weight for Boltz protocol STE.')
flags.DEFINE_float('boltz_ste_soft_end', 0.0, 'Final soft weight for Boltz protocol STE.')
flags.DEFINE_float('boltz_ste_hard_start', 0.0, 'Initial hard weight for Boltz protocol STE.')
flags.DEFINE_float('boltz_ste_hard_end', 1.0, 'Final hard weight for Boltz protocol STE.') 
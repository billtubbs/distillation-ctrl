"""Variable metadata for the distillation column model.

Source: M. Grieve's original Simulink model (August 2020).
"""

# Keyed by the short variable name used throughout the codebase.
# Each entry carries the descriptive name and engineering units.
VAR_INFO = {
    # Manipulated variables (model inputs)
    "V": {"name": "Boilup", "units": "BPH"},
    "D": {"name": "Distillate draw", "units": "BPH"},
    # Controlled variables (model outputs)
    "OHt": {"name": "Overhead temperature", "units": "degF"},
    "L": {"name": "Reflux rate", "units": "BPH"},
    "BmT": {"name": "Bottom temperature", "units": "degF"},
}

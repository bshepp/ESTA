"""Probe subpackage.

The __init__ is deliberately empty so that importing torch-free submodules
(esta.probes.thresholds) does not transitively trigger a torch import via
esta.probes.refusal. Callers should use the full path:

    from esta.probes.thresholds import label_pressure, PressureThresholds
    from esta.probes.refusal import load_refusal_direction, project_activations
"""

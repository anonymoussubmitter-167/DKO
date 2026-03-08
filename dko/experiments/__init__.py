"""
DKO Experiments Module

Contains experiment scripts for:
- Main benchmark evaluation
- Ablation studies
- Sample efficiency analysis (data fraction + conformer count)
- Attention analysis and scaling
- Representation vs architecture study
- Negative control experiments
- 80/20 decomposition study
- SCC validation
- Decision rule calibration
- Sketching experiments
"""

from dko.experiments.main_benchmark import run_main_benchmark
from dko.experiments.decomposition import (
    run_decomposition_study,
    run_full_decomposition_study,
    compute_decomposition,
)

# Alias for backward compatibility
run_decomposition_experiment = run_decomposition_study
from dko.experiments.sample_efficiency import (
    run_sample_efficiency_experiment,
    run_conformer_count_experiment,
    run_full_sample_efficiency_study,
)
from dko.experiments.attention_analysis import (
    run_attention_analysis,
    run_attention_scaling_experiment,
    run_control_scaling_experiment,
)
from dko.experiments.representation_vs_architecture import (
    run_representation_vs_architecture_experiment,
    run_full_rep_vs_arch_study,
)

# Alias for shorter name
run_rep_vs_arch_experiment = run_representation_vs_architecture_experiment
from dko.experiments.negative_controls import (
    run_negative_control_experiment,
    run_scc_advantage_correlation,
)
from dko.experiments.sketching import run_sketching_experiment
from dko.experiments.scc_validation import run_scc_validation
from dko.experiments.decision_rule import run_decision_rule_experiment

__all__ = [
    # Main benchmark
    "run_main_benchmark",
    # Decomposition study
    "run_decomposition_study",
    "run_decomposition_experiment",  # Alias
    "run_full_decomposition_study",
    "compute_decomposition",
    # Sample efficiency
    "run_sample_efficiency_experiment",
    "run_conformer_count_experiment",
    "run_full_sample_efficiency_study",
    # Attention analysis
    "run_attention_analysis",
    "run_attention_scaling_experiment",
    "run_control_scaling_experiment",
    # Representation vs architecture
    "run_representation_vs_architecture_experiment",
    "run_rep_vs_arch_experiment",  # Alias
    "run_full_rep_vs_arch_study",
    # Negative controls
    "run_negative_control_experiment",
    "run_scc_advantage_correlation",
    # Other experiments
    "run_sketching_experiment",
    "run_scc_validation",
    "run_decision_rule_experiment",
]

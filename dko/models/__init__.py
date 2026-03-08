"""
DKO Models Module

Contains implementations of:
- DKO (Distribution Kernel Operators)
- Attention-based aggregation (including Attention-Augmented)
- DeepSets (including DeepSets-Augmented)
- GNN baselines (SchNet, DimeNet++, SphereNet, 3D-Infomax, GEM)
- Ensemble baselines (Mean, Boltzmann, MFA, MIL)
"""

from dko.models.dko import DKO, DKOKernel, DKOFirstOrder, DKONoPSD
from dko.models.dko_variants import (
    DKOEigenspectrum,
    DKOScalarInvariants,
    DKOLowRank,
    DKOGatedFusion,
    DKOResidual,
    DKOCrossAttention,
    DKOSCCRouter,
)
from dko.models.attention import (
    AttentionAggregation,
    MultiHeadAttention,
    AttentionPoolingBaseline,
    AttentionAugmented,
)

# Convenient aliases
AttentionPooling = AttentionAggregation  # Common name alias
from dko.models.deepsets import DeepSets, DeepSetsBaseline, DeepSetsAugmented
from dko.models.gnn_baselines import (
    SchNet,
    DimeNetPP,
    SphereNet,
    ThreeDInfomax,
    GEM,
    get_gnn,
    GNNWithConformerAggregation,
    ConformerAggregation,
    HAS_PYG,
)

# Import PyG versions if available
if HAS_PYG:
    from dko.models.gnn_baselines import SchNetPyG, DimeNetPPPyG, GNNEnsembleWrapper
from dko.models.ensemble_baselines import (
    SingleConformer,
    SingleConformerBaseline,
    MeanFeatureAggregation,
    MultiInstanceLearning,
    MILBaseline,
    MeanEnsemble,
    BoltzmannEnsemble,
    LearnedWeightEnsemble,
)

__all__ = [
    # DKO
    "DKO",
    "DKOKernel",
    "DKOFirstOrder",
    "DKONoPSD",
    # DKO Variants
    "DKOEigenspectrum",
    "DKOScalarInvariants",
    "DKOLowRank",
    "DKOGatedFusion",
    "DKOResidual",
    "DKOCrossAttention",
    "DKOSCCRouter",
    # Attention
    "AttentionAggregation",
    "AttentionPooling",  # Alias for AttentionAggregation
    "MultiHeadAttention",
    "AttentionPoolingBaseline",
    "AttentionAugmented",
    # DeepSets
    "DeepSets",
    "DeepSetsBaseline",
    "DeepSetsAugmented",
    # GNN baselines
    "SchNet",
    "DimeNetPP",
    "SphereNet",
    "ThreeDInfomax",
    "GEM",
    # Ensemble baselines
    "SingleConformer",
    "SingleConformerBaseline",
    "MeanFeatureAggregation",
    "MultiInstanceLearning",
    "MILBaseline",
    "MeanEnsemble",
    "BoltzmannEnsemble",
    "LearnedWeightEnsemble",
]

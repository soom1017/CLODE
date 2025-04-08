# Self attention (clip feature) + Cross attention (score + clip feature) + MLP
from .regressor_v1 import Regressor as Regressor_v1
# CBAM (score + clip feature) + MLP
from .regressor_v2 import Regressor as Regressor_v2
__all__ = ["Regressor_v1", "Regressor_v2"]
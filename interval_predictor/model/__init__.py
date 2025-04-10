# Self attention (clip feature) + Cross attention (score + clip feature) + MLP
from .regressor_v1 import Regressor as Regressor_v1
# CBAM (score + clip feature) + MLP
from .regressor_v2 import Regressor as Regressor_v2
# ECA (score + clip feature) + MLP
from .regressor_v3 import Regressor as Regressor_v3
# ECA (prompt feature + image feature) + MLP
from .regressor_v4 import Regressor as Regressor_v4
# Cross attention (prompt feature + image feature) + MLP
from .regressor_v5 import Regressor as Regressor_v5
# CBAM (prompt feature + image feature) + MLP
from .regressor_v6 import Regressor as Regressor_v6
__all__ = ["Regressor_v1", "Regressor_v2", "Regressor_v3", "Regressor_v4", "Regressor_v5", "Regressor_v6"]
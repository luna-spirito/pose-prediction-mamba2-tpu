"""stickytape entry, look into training.py"""
# from __future__ import annotations

try:
    # When imported as move2.modules
    from move2.modules import (
        constants,
        dataset_utils,
        mamba2,
        model,
        training,
        visualization,
    )

    # Re-export main functions
    from move2.modules.dataset_utils import prepare_dataset
    from move2.modules.training import train
    from move2.modules.visualization import visualize
except ImportError:
    # When run directly
    import constants
    import dataset_utils
    import mamba2
    import model
    import training
    import visualization

    # Re-export main functions
    from dataset_utils import prepare_dataset
    from training import train
    from visualization import visualize

__all__ = ["prepare_dataset", "train", "visualize"]

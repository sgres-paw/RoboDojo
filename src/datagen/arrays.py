"""Array conversion shared across the datagen pipeline."""

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Convert a simulator value to a numpy array."""
    # get_instance_pose returns torch for rigid bodies and numpy for everything else.
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value)

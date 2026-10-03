"""Small runtime adapters for the legacy Gym/PyFlex SoftGym code."""

from __future__ import annotations

import sys
import types


def install_legacy_compat() -> None:
    """Expose Gymnasium under the legacy Gym names used by SoftGym."""

    import numpy as np

    # SoftGym still uses aliases removed in NumPy 1.24.
    for name, value in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)

    if "gym" in sys.modules:
        return
    import gymnasium as gym

    gym.error = types.SimpleNamespace(DependencyNotInstalled=ImportError)
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = gym.spaces

    # SoftGym imports this only for optional video export. Training does not
    # record videos, so avoid importing the legacy moviepy stack.
    if "softgym.utils.visualization" not in sys.modules:
        visualization = types.ModuleType("softgym.utils.visualization")
        visualization.save_numpy_as_gif = lambda *args, **kwargs: None
        sys.modules["softgym.utils.visualization"] = visualization

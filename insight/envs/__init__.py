"""Toy environments. Importing the package registers them with stable-envs."""

from insight.envs.moving_dot import MovingDot
from insight.envs.three_dots import ThreeDots

__all__ = ["MovingDot", "ThreeDots"]

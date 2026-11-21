"""Von Mises heading policy of the moving dot, after Sobal et al. (2022, arXiv 2211.10831), App. A.2."""

import numpy as np
import torch


class VonMisesHeading:
    """Open-loop action sequences of a dot that keeps roughly one heading.

    Each sequence draws a heading uniformly. Each action's direction is Von Mises around that heading
    and its norm is uniform in ``[0, max_step]``. The authors' code uses concentration 0.2, the paper
    1.3.
    """

    def __init__(self, concentration=0.2):
        self.concentration = concentration

    def __call__(self, n, steps, max_step, rng):
        """``(n, steps, 2)`` float32 actions from the numpy generator ``rng``."""
        heading = rng.uniform(0, 2 * np.pi, size=(n, 1))
        theta = rng.vonmises(np.broadcast_to(heading, (n, steps)), self.concentration)
        norm = rng.uniform(0, max_step, size=(n, steps, 1))
        return torch.from_numpy(
            (norm * np.stack([np.cos(theta), np.sin(theta)], -1)).astype(np.float32)
        )

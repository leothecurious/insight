import numpy as np
import pytest
import torch

from insight.metrics.pairs import pair_cosines


def test_pair_cosines_match_tag_after_a_three_way_swap():
    """Using the forward index for same-tag pairs compares different tags in a three-cycle."""
    own = torch.eye(3)
    partner = np.array([1, 2, 0])
    assert pair_cosines(own, own[[1, 2, 0]], partner) == pytest.approx(
        {"same_tag": 1.0, "same_content": -0.5, "baseline": -0.5}
    )

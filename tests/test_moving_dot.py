import stable_envs as snv
from stable_envs.testing import conformance

import insight.envs  # noqa: F401


def test_moving_dot_conforms_to_the_renderer_checks():
    """The checks stable-envs runs on its own classes: a frame that depended on the batch it was drawn
    in, on hidden state, or that ignored a coordinate would pass silently through training."""
    with snv.make("MovingDot", num_envs=4, device="cpu") as env:
        assert conformance.check_batch_invariance(env)["max_abs_diff"] == 0
        assert conformance.check_determinism(env)["identical"]
        assert conformance.check_state_dims_matter(env)["dead_dims"] == []

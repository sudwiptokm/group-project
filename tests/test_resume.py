"""Training has to survive being killed partway.

Long background runs on this machine are terminated at ~53 minutes. A 30k-step
PPO run takes ~68, so it can never finish in one go, and SB3 only writes a
checkpoint when learn() returns -- one kill lost 85% of a completed run.

So training checkpoints periodically and resumes from what is on disk, and the
budget is counted in TOTAL steps rather than steps-per-invocation: three
resumed chunks of a 30k run must train 30k, not 90k, or the arm gets a bigger
budget than the one it is being compared against.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_common import remaining_steps  # noqa: E402


class TestRemainingSteps:
    def test_a_fresh_run_trains_the_whole_budget(self):
        assert remaining_steps(30_000, 0) == 30_000

    def test_a_resumed_run_trains_only_the_shortfall(self):
        # the killed PPO seed-0 run had ~24k of 30k steps done
        assert remaining_steps(30_000, 24_000) == 6_000

    def test_a_finished_run_trains_nothing(self):
        assert remaining_steps(30_000, 30_000) == 0

    def test_an_overshooting_checkpoint_does_not_go_negative(self):
        # SB3 rounds up to a whole rollout, so num_timesteps can exceed target
        assert remaining_steps(30_000, 30_512) == 0

    @pytest.mark.parametrize("done", [0, 1, 29_999, 30_000, 50_000])
    def test_never_exceeds_the_budget(self, done):
        assert 0 <= remaining_steps(30_000, done) <= 30_000

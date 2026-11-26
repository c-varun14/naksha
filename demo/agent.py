"""Live ablation for the policy's recurrent state."""

import training.paths  # noqa: F401
from submission.agent import DoomAgent


class MemoryOffAgent(DoomAgent):
    def act(self, obs):
        self.state = self.model.initial_state(1)
        return super().act(obs)

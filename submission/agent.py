"""Evaluation entry point for the recurrent policy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Space

SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR))

from naksha import actions, features, model  # noqa: E402


class DoomAgent:
    def __init__(
        self, observation_space: Space, action_space: Space, device: str,
        *, weights: str | Path | None = None,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        torch.set_num_threads(1)
        path = weights or os.environ.get("NAKSHA_WEIGHTS") or SUBMISSION_DIR / "weights" / "naksha.pt"
        self.model, _ = model.load_checkpoint(path)
        self.model.eval()
        zeros = {key: torch.from_numpy(np.zeros((1, 1, *shape), dtype))
                 for key, (shape, dtype) in features.SHAPES.items()}
        with torch.inference_mode():
            self.model(zeros, self.model.initial_state(1))
        self.reset()

    def reset(self) -> None:
        self.state = self.model.initial_state(1)
        self.prev_action = 0
        self.last_out: dict[str, np.ndarray | float] = {}

    @torch.inference_mode()
    def act(self, obs: dict) -> np.ndarray:
        encoded = features.encode(obs, self.prev_action)
        inputs = {key: torch.from_numpy(data)[None, None] for key, data in encoded.items()}
        out, self.state = self.model(inputs, self.state)
        logits = out["logits"][0, 0]
        self.last_out = {
            "logits": logits.numpy(),
            "intention": out["intention"][0, 0].softmax(dim=-1).numpy(),
            "exit_vis": out["exit_vis"][0, 0].sigmoid().item(),
        }
        self.prev_action = int(logits.argmax())
        return actions.to_buttons(self.prev_action)

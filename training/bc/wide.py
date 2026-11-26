"""Double-width recurrent policy for behavior cloning."""

from __future__ import annotations

import torch
from torch import Tensor, nn

import training.paths  # noqa: F401
from naksha.actions import NUM_ACTIONS
from naksha.model import INTENTIONS, MemoryModule


class WideNet(nn.Module):
    """Recurrent policy with twice the internal width of NakshaNet.

    >>> print(WideNet().count_params())
    6578775
    """

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 512
        entity_width = 128
        self.rays = nn.Sequential(
            nn.Conv1d(7, 64, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 64, 512), nn.ReLU(),
        )
        self.ents = nn.Sequential(
            nn.Linear(17, entity_width), nn.ReLU(), nn.Linear(entity_width, entity_width), nn.ReLU(),
        )
        self.entity_query = nn.Parameter(torch.randn(entity_width))
        self.empty_entity = nn.Parameter(torch.zeros(1, 1, entity_width))
        self.vitals = nn.Sequential(nn.Linear(17, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.prev_action = nn.Linear(NUM_ACTIONS, 64)
        self.pixels = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 2 * 3, 256), nn.ReLU(),
        )
        self.core = nn.Sequential(nn.Linear(512 + entity_width + 128 + 64 + 256, self.hidden_size), nn.ReLU())
        self.memory = MemoryModule()
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.logits = nn.Linear(self.hidden_size, NUM_ACTIONS)
        self.value = nn.Linear(self.hidden_size, 1)
        self.intention = nn.Linear(self.hidden_size, len(INTENTIONS))
        self.exit_vis = nn.Linear(self.hidden_size, 1)
        self.exit_dir = nn.Linear(self.hidden_size, 2)

    def initial_state(self, batch: int, device: str | torch.device = "cpu") -> Tensor:
        return torch.zeros(1, batch, self.hidden_size, device=device, dtype=self.gru.weight_hh_l0.dtype)

    def forward(self, inputs: dict[str, Tensor], state: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        time, batch = inputs["vitals"].shape[:2]
        rays = self.rays(inputs["rays"].flatten(0, 1).transpose(1, 2))
        ents = self.ents(inputs["ents"].flatten(0, 1))
        ents = torch.cat((ents, self.empty_entity.expand(time * batch, -1, -1)), dim=1)
        mask = inputs["ent_mask"].flatten(0, 1).bool()
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        scores = (ents @ self.entity_query) * (self.entity_query.numel() ** -0.5)
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), ents).squeeze(1)

        vitals = self.vitals(inputs["vitals"].flatten(0, 1))
        prev_action = self.prev_action(inputs["prev_action"].flatten(0, 1))
        pixels = self.pixels(inputs["pixels"].flatten(0, 1).float() / 255)
        z = self.core(torch.cat((rays, pooled, vitals, prev_action, pixels), dim=-1))
        z, _ = self.memory(z.reshape(time, batch, self.hidden_size), None)
        z, state = self.gru(z, state)
        return {
            "logits": self.logits(z),
            "value": self.value(z).squeeze(-1),
            "intention": self.intention(z),
            "exit_vis": self.exit_vis(z).squeeze(-1),
            "exit_dir": self.exit_dir(z).tanh(),
        }, state

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

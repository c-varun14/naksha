"""Recurrent policy shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .actions import ACTIONS_VERSION, NUM_ACTIONS
from .features import FEATURES_VERSION

MODEL_VERSION = 1
HIDDEN = 256
INTENTIONS = ("explore", "get_key", "go_door", "go_exit", "fight")
_VERSIONS = {
    "FEATURES_VERSION": FEATURES_VERSION,
    "ACTIONS_VERSION": ACTIONS_VERSION,
    "MODEL_VERSION": MODEL_VERSION,
}


class MemoryModule(nn.Module):
    def forward(self, z: Tensor, mem_state: Tensor | None) -> tuple[Tensor, Tensor | None]:
        return z, mem_state


class NakshaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rays = nn.Sequential(
            nn.Conv1d(7, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 64, 256), nn.ReLU(),
        )
        self.ents = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.entity_query = nn.Parameter(torch.randn(64))
        self.empty_entity = nn.Parameter(torch.zeros(1, 1, 64))
        self.vitals = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.prev_action = nn.Linear(NUM_ACTIONS, 32)
        self.pixels = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 2 * 3, 128), nn.ReLU(),
        )
        self.core = nn.Sequential(nn.Linear(256 + 64 + 64 + 32 + 128, HIDDEN), nn.ReLU())
        self.memory = MemoryModule()
        self.gru = nn.GRU(HIDDEN, HIDDEN)
        self.logits = nn.Linear(HIDDEN, NUM_ACTIONS)
        self.value = nn.Linear(HIDDEN, 1)
        self.intention = nn.Linear(HIDDEN, len(INTENTIONS))
        self.exit_vis = nn.Linear(HIDDEN, 1)
        self.exit_dir = nn.Linear(HIDDEN, 2)

    def initial_state(self, batch: int, device: str | torch.device = "cpu") -> Tensor:
        return torch.zeros(1, batch, HIDDEN, device=device, dtype=self.gru.weight_hh_l0.dtype)

    def forward(self, inputs: dict[str, Tensor], state: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        time, batch = inputs["vitals"].shape[:2]
        rays = self.rays(inputs["rays"].flatten(0, 1).transpose(1, 2))
        ents = self.ents(inputs["ents"].flatten(0, 1))
        ents = torch.cat((ents, self.empty_entity.expand(time * batch, -1, -1)), dim=1)
        mask = inputs["ent_mask"].flatten(0, 1).bool()
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        scores = (ents @ self.entity_query) * (64 ** -0.5)
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), ents).squeeze(1)

        vitals = self.vitals(inputs["vitals"].flatten(0, 1))
        prev_action = self.prev_action(inputs["prev_action"].flatten(0, 1))
        pixels = self.pixels(inputs["pixels"].flatten(0, 1).float() / 255)
        z = self.core(torch.cat((rays, pooled, vitals, prev_action, pixels), dim=-1))
        z, _ = self.memory(z.reshape(time, batch, HIDDEN), None)
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


class EnsembleNet(nn.Module):
    """Several trained policies voting together.

    Each member keeps its own recurrent state. Every step the members' action
    probabilities are averaged and the result is returned as log-probabilities,
    so argmax or sampling over it works exactly as for one model.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.members = nn.ModuleList(NakshaNet() for _ in range(size))

    def initial_state(self, batch: int, device: str | torch.device = "cpu") -> Tensor:
        return torch.stack([member.initial_state(batch, device) for member in self.members])

    def forward(self, inputs: dict[str, Tensor], state: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        outs, states = [], []
        for member, member_state in zip(self.members, state):
            out, new_state = member(inputs, member_state)
            outs.append(out)
            states.append(new_state)

        def mean_probs(key: str) -> Tensor:
            return torch.stack([out[key].softmax(dim=-1) for out in outs]).mean(0).clamp_min(1e-8).log()

        exit_vis = torch.stack([out["exit_vis"].sigmoid() for out in outs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return {
            "logits": mean_probs("logits"),
            "value": torch.stack([out["value"] for out in outs]).mean(0),
            "intention": mean_probs("intention"),
            "exit_vis": torch.logit(exit_vis),
            "exit_dir": torch.stack([out["exit_dir"] for out in outs]).mean(0),
        }, torch.stack(states)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[NakshaNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    meta = checkpoint.get("meta", {})
    for name, expected in _VERSIONS.items():
        if meta.get(name) != expected:
            raise ValueError(f"{name} mismatch: expected {expected}, got {meta.get(name)!r}")
    model = NakshaNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, meta


def save_checkpoint(path: str | Path, model: NakshaNet, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": {**meta, **_VERSIONS}}, path)

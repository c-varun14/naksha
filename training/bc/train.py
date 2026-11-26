"""Train the recurrent policy by behavior cloning."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

import training.paths  # noqa: F401
from naksha import features, model
from training.bc.wide import WideNet
from training.data.augment import mirror_rate_batch
from training.data.shards import SequenceDataset

BURN_IN = 16


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return values.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)


def _batch_loss(net: model.NakshaNet | WideNet, chunk: dict[str, np.ndarray],
                device: torch.device) -> tuple[Tensor, Tensor]:
    inputs = {key: torch.from_numpy(chunk[key]).to(device) for key in features.KEYS}
    state = net.initial_state(chunk["valid"].shape[1], device)
    with torch.no_grad():
        _, state = net({key: data[:BURN_IN] for key, data in inputs.items()}, state)
    out, _ = net({key: data[BURN_IN:] for key, data in inputs.items()}, state)

    targets = {key: torch.from_numpy(chunk[key][BURN_IN:]).to(device)
               for key in ("valid", "label", "intention", "exit_vis", "exit_dir")}
    valid = targets["valid"]
    label = targets["label"].long()
    labelled = valid & (label >= 0)
    action_ce = F.cross_entropy(out["logits"].flatten(0, 1), label.masked_fill(~labelled, 0).flatten(),
                                reduction="none").reshape_as(label)
    intention = targets["intention"].long().masked_fill(~labelled, 0)
    intention_ce = F.cross_entropy(out["intention"].flatten(0, 1), intention.flatten(),
                                   reduction="none").reshape_as(label)
    visible = targets["exit_vis"]
    exit_bce = F.binary_cross_entropy_with_logits(out["exit_vis"], visible.float(), reduction="none")
    visible_mask = valid & (visible == 1)
    direction = targets["exit_dir"].float().masked_fill(~visible_mask.unsqueeze(-1), 0)
    exit_mse = (out["exit_dir"] - direction).square().mean(dim=-1)
    loss = (_masked_mean(action_ce, labelled) + 0.2 * _masked_mean(intention_ce, labelled)
            + 0.1 * _masked_mean(exit_bce, valid) + 0.1 * _masked_mean(exit_mse, visible_mask))
    accuracy = _masked_mean(out["logits"].argmax(dim=-1) == label, labelled)
    return loss, accuracy


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=training.paths.ROOT, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    train_data, val_data = SequenceDataset(args.data).split(0.1, seed=0, min_steps=BURN_IN + 1)
    if not train_data.lengths or not val_data.lengths:
        raise ValueError(f"training requires two episodes with at least {BURN_IN + 1} steps each")
    rng = np.random.default_rng()
    chunk_len = args.seq + BURN_IN
    val_chunk = val_data.sample(args.batch, chunk_len, np.random.default_rng(1), min_steps=BURN_IN + 1)
    wide = getattr(args, "wide", False)
    if wide:
        net = WideNet().to(device)
    else:
        net = model.load_checkpoint(args.init, device)[0] if args.init else model.NakshaNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("runs") / args.out.stem / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = {"data": args.data, "git_sha": _git_sha(), "val_loss": float("inf"), "mirror": args.mirror}

    def validate(update: int) -> tuple[float, float]:
        net.eval()
        with torch.inference_mode():
            loss, acc = (float(value) for value in _batch_loss(net, val_chunk, device))
        if loss < best["val_loss"]:
            best.update(updates=update, val_loss=loss, val_acc=acc)
            if wide:
                torch.save({"state_dict": net.state_dict(), "meta": {**best, "arch": "wide2"}}, args.out)
            else:
                model.save_checkpoint(args.out, net, best)
        net.train()
        return loss, acc

    updates = 0
    train_metrics: tuple[float | str, float | str] = ("", "")
    with log_path.open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(("update", "loss", "train_acc", "val_loss", "val_acc"))
        deadline = time.monotonic() + args.minutes * 60
        while time.monotonic() < deadline:
            chunk = train_data.sample(args.batch, chunk_len, rng, min_steps=BURN_IN + 1)
            if args.mirror:
                chunk = mirror_rate_batch(chunk, args.mirror, rng)
            optimizer.zero_grad(set_to_none=True)
            loss, acc = _batch_loss(net, chunk, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            updates += 1
            if updates % 200 == 0:
                train_metrics = (float(loss.detach()), float(acc))
                val_metrics = validate(updates) if updates % 1_000 == 0 else ("", "")
                writer.writerow((updates, *train_metrics, *val_metrics))
                log.flush()
        if updates == 0 or updates % 1_000:
            val_metrics = validate(updates)
            if updates % 200:
                writer.writerow((updates, float(loss.detach()), float(acc), *val_metrics))
            elif updates == 0:
                writer.writerow((updates, *train_metrics, *val_metrics))
    return {**best, "total_updates": updates, "out": str(args.out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mirror", type=float, default=0.0)
    args = parser.parse_args()
    if args.wide and args.init:
        parser.error("--wide cannot be combined with --init")
    if not 0 <= args.mirror <= 1:
        parser.error("mirror must be between 0 and 1")
    if not np.isfinite(args.minutes) or args.minutes < 0 or min(args.batch, args.seq) < 1:
        parser.error("minutes must be finite and nonnegative, and batch and seq must be positive")
    print(json.dumps(train(args)))


if __name__ == "__main__":
    main()

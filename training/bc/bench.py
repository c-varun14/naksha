"""Benchmark single-step recurrent policy inference on one CPU thread."""

import argparse
import json
from time import perf_counter

import numpy as np
import torch

import training.paths  # noqa: F401
from naksha import features
from naksha.model import NakshaNet
from training.bc.wide import WideNet


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")

    torch.set_num_threads(1)
    net = (WideNet() if args.wide else NakshaNet()).eval()
    arrays = {key: np.zeros((1, 1, *shape), dtype) for key, (shape, dtype) in features.SHAPES.items()}
    inputs = {key: torch.from_numpy(arrays[key]) for key in features.KEYS}
    state = net.initial_state(1)
    for _ in range(20):
        _, state = net(inputs, state)

    latencies = np.empty(args.n)
    for i in range(args.n):
        start = perf_counter()
        _, state = net(inputs, state)
        latencies[i] = (perf_counter() - start) * 1_000
    p50, p99 = np.percentile(latencies, [50, 99])
    print(json.dumps({"arch": "wide2" if args.wide else "naksha", "params": net.count_params(),
                      "p50_ms": p50, "p99_ms": p99}))


if __name__ == "__main__":
    main()

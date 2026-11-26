import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402, F401
from naksha import features, model  # noqa: E402
from naksha.memory import (  # noqa: E402
    CROP_SHAPE, MEMORY_VERSION, GridMemory, NakshaMemNet, load_any, pack_crops, unpack_crops,
)
from training.bc.train_mem import MemoryDataset, _batch_loss, _build_model  # noqa: E402
from training.data.shards import _FIELDS, write_episode  # noqa: E402
from training.memory import odometry_train, precompute  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _corridor() -> tuple[np.ndarray, ...]:
    angles = np.arange(64) * (2 * np.pi / 64)
    lateral = np.abs(np.sin(angles))
    distances = 100 / np.maximum(lateral, 100 / 1024) / 1024
    return (distances, (distances < 1).astype(int), np.zeros((32, 5)),
            np.zeros(32, dtype=int), np.zeros(32, dtype=bool))


def test_corridor_and_rotation() -> None:
    grid = GridMemory()
    rays, hit, entities, kinds, mask = _corridor()
    entities[0] = (0.1, 0, 1, 0, 0)
    kinds[0] = 9
    mask[0] = True
    hit[16] = 4
    grid.step(rays, hit, entities, kinds, mask, (0, 0, 0))
    assert grid.crops()[5, 12, 9]
    assert grid.crops()[7, 9, 12]
    hit[16] = 1
    mask[0] = False
    grid.step(rays, hit, entities, kinds, mask, (32, 0, 0))
    crop = grid.crops()
    assert crop[1, 12, 9] and crop[1, 12, 16]
    assert crop[0, 10, 12] and crop[6, 12, 12]
    assert not crop[5].any() and not crop[13].any()
    assert crop[7].any()
    np.testing.assert_array_equal(grid._coarse, grid.grid.reshape(8, 48, 4, 48, 4).max(axis=(2, 4)))

    grid.step(rays, hit, entities, kinds, mask, (0, 0, 90))
    assert grid.crops()[7, 12, 14]
    for _ in range(3):
        grid.step(rays, hit, entities, kinds, mask, (0, 0, 90))
    assert grid.pose() == (32.0, 0.0, 0.0)
    rgb = grid.render_rgb(2)
    assert rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
    grid.reset()
    assert not grid.crops().any()
    assert grid.pose() == (0.0, 0.0, 0.0)


def test_crop_packing_and_step_speed() -> None:
    grid = GridMemory()
    rays, hit, ents, kinds, mask = _corridor()
    rays[::2] = 1
    samples = []
    for _ in range(200):
        start = perf_counter_ns()
        grid.step(rays, hit, ents, kinds, mask, (1.0, 0.25, 3.5))
        crop = grid.crops()
        samples.append(perf_counter_ns() - start)
    elapsed = median(samples) / 1e6
    print(f"Grid step + crops median {elapsed:.3f} ms")
    assert elapsed < 1.5
    assert crop.shape == CROP_SHAPE and crop.dtype == np.uint8
    assert crop.max() == 1
    batch = np.stack((crop, 1 - crop))
    packed = pack_crops(batch)
    assert packed.shape == (2, 16, 24, 3) and packed.dtype == np.uint8
    np.testing.assert_array_equal(unpack_crops(packed), batch)


@pytest.mark.parametrize("memory", (False, True))
def test_load_any_and_forward(tmp_path: Path, memory: bool) -> None:
    net = NakshaMemNet() if memory else model.NakshaNet()
    path = tmp_path / "policy.pt"
    model.save_checkpoint(path, net, {"memory": MEMORY_VERSION} if memory else {})
    restored, meta = load_any(path)
    assert type(restored) is type(net)
    assert meta.get("memory") == (MEMORY_VERSION if memory else None)
    for key, expected in net.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)

    inputs = {key: torch.from_numpy(np.zeros((3, 2, *shape), dtype))
              for key, (shape, dtype) in features.SHAPES.items()}
    if memory:
        inputs["memory"] = torch.ones((3, 2, *CROP_SHAPE), dtype=torch.uint8)
        assert sum(p.numel() for p in net.odometry.parameters()) < 150_000
        assert net.count_params() < 2_500_000
    with torch.inference_mode():
        output, state = restored(inputs, restored.initial_state(2))
    assert {key: value.shape for key, value in output.items()} == {
        "logits": (3, 2, 14), "value": (3, 2), "intention": (3, 2, 5),
        "exit_vis": (3, 2), "exit_dir": (3, 2, 2),
    }
    assert state.shape == (1, 2, model.HIDDEN)
    assert all(torch.isfinite(value).all() for value in output.values())
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["meta"]["MODEL_VERSION"] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="MODEL_VERSION"):
        load_any(path)


def test_odometry_targets_and_drift() -> None:
    pose = np.array([[0, 0, 90], [-2, 10, 76], [-2, 10, 359], [-2, 10, 3]], np.float32)
    np.testing.assert_allclose(odometry_train.motion_targets(pose),
                               [[10, 2, -14], [0, 0, -77], [0, 0, 4]], atol=1e-5)
    pose = np.column_stack((np.arange(201) * 10, np.zeros(201), np.zeros(201)))
    motion = odometry_train.motion_targets(pose)
    position, heading = odometry_train._window_drift(motion, pose)
    np.testing.assert_allclose((position, heading), [[0], [0]])


def test_odometry_training_and_memory_bc_smoke(tmp_path: Path) -> None:
    directory = tmp_path / "episodes"
    for i in range(2):
        steps = {key: np.zeros((18, *shape), dtype) for key, (shape, dtype) in _FIELDS.items()}
        steps["rays"][:, :, :2] = 1
        steps["prev_action"][:, 1] = 1
        steps["pose"][:, 0] = 10 * np.arange(18)
        steps["vitals"][:, 14] = 0.1
        write_episode(directory, f"episode{i}", steps, {})

    path = tmp_path / "odometry.pt"
    stats = odometry_train.train([directory], minutes=1, out=path, max_updates=1, batch_size=4)
    assert path.is_file() and stats["updates"] == 1 and stats["held_out_episodes"] == 1
    assert np.isfinite(stats["learned"]["mae_fwd"])
    shard = directory / "episode0.npz"
    original = shard.read_bytes()
    assert precompute.precompute([directory], path, procs=1) > 0
    assert shard.read_bytes() == original
    with np.load(precompute.sidecar_path(shard)) as archive:
        assert archive["memory_bits"].shape == (18, 16, 24, 3)

    baseline = model.NakshaNet()
    init = tmp_path / "base.pt"
    model.save_checkpoint(init, baseline, {})
    device = torch.device("cpu")
    net = _build_model(init, path, device)
    torch.testing.assert_close(net.core[0].weight[:, :baseline.core[0].in_features], baseline.core[0].weight)
    assert not any(param.requires_grad for param in net.odometry.parameters())

    dataset = MemoryDataset([directory])
    train, _ = dataset.split()
    chunk = dataset.sample(train, 1, 20, np.random.default_rng(0))
    assert chunk["memory"].shape == (20, 1, *CROP_SHAPE)
    assert chunk["memory"][chunk["valid"]].any()
    assert not chunk["memory"][~chunk["valid"]].any()
    loss, accuracy = _batch_loss(net, chunk, device)
    assert torch.isfinite(loss) and torch.isfinite(accuracy)
    loss.backward()
    assert net.memory_encoder[0].weight.grad is not None

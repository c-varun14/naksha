"""Keep the official game and an HD spectator in lockstep."""

from contextlib import ExitStack, contextmanager
import logging
from types import FunctionType, SimpleNamespace

import vizdoom as vd

import training.paths  # noqa: F401
from doom_env import make_env
from doom_env.wrapper import DoomEnv


class _HDGame(vd.DoomGame):
    def set_screen_resolution(self, resolution):
        super().set_screen_resolution(vd.ScreenResolution.RES_640X480)


class HDDoomEnv(DoomEnv):
    # The official constructor has no game factory hook. Give it a private
    # ViZDoom binding so all its settings and initialization order stay intact.
    __init__ = FunctionType(
        DoomEnv.__init__.__code__,
        {**DoomEnv.__init__.__globals__, "vd": SimpleNamespace(**(vars(vd) | {"DoomGame": _HDGame}))},
        argdefs=DoomEnv.__init__.__defaults__,
    )
    __init__.__kwdefaults__ = DoomEnv.__init__.__kwdefaults__


@contextmanager
def open_games(wad, map_name="MAP01", *, episode_ticks):
    with ExitStack() as stack:
        official = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
        stack.callback(official.close)
        spectator = HDDoomEnv(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
        stack.callback(spectator.close)
        yield official, spectator


class Lockstep:
    def __init__(self, official, spectator):
        self.official = official
        self.spectator = spectator
        self.desync_step = None
        self.step_number = 0
        self.pose = None

    def reset(self, *, seed=0):
        self.desync_step = None
        self.step_number = 0
        obs, _ = self.official.reset(seed=seed)
        self.spectator.reset(seed=seed)
        self._compare((False, False), (False, False))
        return obs

    def step(self, action):
        self.step_number += 1
        obs, _, terminated, truncated, _ = self.official.step(action)
        _, _, spec_terminated, spec_truncated, _ = self.spectator.step(action)
        self._compare((terminated, truncated), (spec_terminated, spec_truncated))
        return obs, terminated, truncated, spec_terminated or spec_truncated

    def _compare(self, flags, spectator_flags):
        self.pose = official = self.official.privileged_state()
        spectator = self.spectator.privileged_state()
        differences = {}
        for name in ("x", "y", "angle", "health", "kills", "finished", "dead", "timeout", "exit_reached"):
            a, b = official[name], spectator[name]
            if name in ("x", "y"):
                match = abs(a - b) <= 0.01
            elif name == "angle":
                match = abs((a - b + 180) % 360 - 180) <= 0.01
            else:
                match = a == b
            if not match:
                differences[name] = (a, b)
        if flags != spectator_flags:
            differences["terminated/truncated"] = (flags, spectator_flags)
        if differences and self.desync_step is None:
            self.desync_step = self.step_number
            logging.getLogger(__name__).warning(
                "Spectator desync at step %d (official, spectator): %s",
                self.desync_step, differences,
            )

    def frame(self, obs):
        return self.spectator.render() if self.desync_step is None else obs["rgb"]

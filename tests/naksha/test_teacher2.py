from doom_env import make_env

from training import paths  # noqa: F401
from naksha.actions import NUM_ACTIONS, to_buttons
from training.gen.compile import write_map
from training.teacher.teacher2 import Teacher2


def test_teacher2_finishes_small_arena(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    grid = (
        "#########\n"
        "#....z..#\n"
        "#P.....E#\n"
        "#....z..#\n"
        "#########"
    )
    wad = write_map(grid, tmp_path / "arena.wad", player_angle=0)
    env = make_env(str(wad), episode_ticks=700, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        teacher = Teacher2({"grid": grid}, seed=0)
        teacher.reset(env)
        while True:
            macro, _, _ = teacher.act(obs, env.privileged_state())
            assert type(macro) is int and 0 <= macro < NUM_ACTIONS
            obs, _, terminated, truncated, _ = env.step(to_buttons(macro))
            if terminated or truncated:
                break
    finally:
        env.close()

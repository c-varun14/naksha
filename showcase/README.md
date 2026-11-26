# Showcase maps

- `01-door.txt` shows an unlocked door across a straight corridor.
- `02-remember-the-key.txt` shows a red door straight ahead and a key at the back of a dead-end side room. Return through the same passage to unlock the door.
- `03-two-keys.txt` puts the blue key beyond the red door and the exit beyond the blue door.
- `04-ambush.txt` puts three zombiemen and an imp in the room leading to the exit.
- `05-loop.txt` offers upper and lower routes around a central wall.
- `06-maze-simple.txt` winds through a 25 × 25 grid with three dead ends and cleared all five trial seeds.
- `06-maze.txt` adds more branches and turns to the same footprint and timed out on all five trial seeds.
- `07-judges-choice.txt` gives judges a small room to edit in the demo.

From the repository root, run:

```bash
uv run --no-sync python -m demo.server --showcase showcase --weights submission/weights/naksha.pt
```

Open <http://127.0.0.1:8765>, select a map from the examples, and press **Play**.
Episodes have a 2,100-tic budget, about 60 seconds at 1× speed and at most 525 decisions.

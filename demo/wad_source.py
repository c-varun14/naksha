"""Validate and stage a local map file for the demo."""

import os
from pathlib import Path
import re
import shutil
import stat

import training.paths  # noqa: F401
from doom_env.wad import list_maps, map_lumps, read_wad
from doom_env.wrapper import IWAD

WAD_SHORTCUTS = [
    {"id": f"Freedoom {name}", "wad": IWAD, "map": name}
    for name in ("MAP01", "MAP02")
]


def stage_wad(path, map_name, destination):
    if not isinstance(path, (str, Path)) or not Path(path).is_absolute():
        raise ValueError("wad must be an absolute filesystem path")
    source = Path(path)
    if source.suffix.lower() != ".wad":
        raise ValueError("wad path must end in .wad")
    if not isinstance(map_name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,7}", map_name):
        raise ValueError("map must be a WAD map name such as MAP01 or E1M1")
    map_name = map_name.upper()

    # Read only the selected file and give both games a validated copy with a fixed name.
    try:
        with open(source, "rb", opener=_open_regular_file) as file, destination.open("wb") as copy:
            shutil.copyfileobj(file, copy)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read WAD file {source}: {exc}") from exc
    try:
        lumps = read_wad(destination)
    except ValueError as exc:
        reason = str(exc).removeprefix(f"{destination}: ")
        raise ValueError(f"Invalid WAD file {source}: {reason}") from exc
    if map_name not in list_maps(lumps):
        raise ValueError(f"WAD file {source} does not contain map {map_name}")
    level = map_lumps(lumps, map_name)
    if not level.get("TEXTMAP") and not {"THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SECTORS"} <= level.keys():
        raise ValueError(f"WAD file {source} has incomplete map data for {map_name}")
    return map_name


def _open_regular_file(path, flags):
    fd = os.open(path, flags | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("expected a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd

"""
Folders of this computer, from <root>/paths.json: optional, not in git (see paths.example.json).

In WSL, the trainings read and write only the Linux disk: a file of /mnt/c costs ~10 ms to create
or to read back from WSL, ~0.1 ms on the Linux disk (see bench_scryfall.py). webcam.py, on Windows,
keeps the defaults (<root>/scryfall and <root>/runs): every model trained in WSL is published there
(see publish). Example:
    {"linux": {"scryfall_dir": "~/scryfall", "runs_dir": "~/mtg_runs"}}

Keys, all optional, in a "linux" or "windows" section (that system only) or at the top level (every
system). '~' is the home folder, a relative path starts from <root>.
- scryfall_dir: the cards, <dir>/exp_<code>/info.json and <dir>/exp_<code>/image/*.jpg.
  Default <root>/scryfall. It must exist: a wrong path would download every image again elsewhere.
- runs_dir: what the trainings write (keys, datasets, check images, Ultralytics runs, models).
  Default <root>/runs. Created if missing.
The MTG_PATHS_FILE environment variable replaces <root>/paths.json (ex: to test another setting).

Check the folders used: python3 src/local_paths.py
"""
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# Project root (<root>), independent of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
PATHS_FILE = Path(os.environ.get("MTG_PATHS_FILE", ROOT / "paths.json"))
SYSTEM = "windows" if sys.platform == "win32" else "linux"
# Where webcam.py (Windows) looks for the models: <root>/runs.
PUBLISH_DIR = ROOT / "runs"


def _load(paths_file: Path) -> Dict[str, str]:
    """The settings of paths_file for this system: its top level, then its "linux" / "windows" section."""
    if not paths_file.exists():
        return {}
    with open(paths_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    values = {name: value for name, value in data.items() if name not in ("linux", "windows")}
    values.update(data.get(SYSTEM, {}))
    return values


def _folder(values: Dict[str, str], name: str, default: Path, create: bool) -> Path:
    """Folder `name` of the settings, or `default`. Created if `create`, else it must exist."""
    if name not in values:
        return default
    path = Path(os.path.expanduser(values[name]))
    path = path if path.is_absolute() else ROOT / path
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise FileNotFoundError(f"'{name}' of {PATHS_FILE} doesn't exist: {path}. "
                                f"Copy the files there first, or fix {PATHS_FILE.name}.")
    return path


_settings = _load(PATHS_FILE)
SCRYFALL_DIR = _folder(_settings, "scryfall_dir", ROOT / "scryfall", create=False)
RUNS_DIR = _folder(_settings, "runs_dir", ROOT / "runs", create=True)


def publishing() -> bool:
    """True if the trainings write somewhere else than <root>/runs: their models are then published there."""
    return RUNS_DIR.resolve() != PUBLISH_DIR.resolve()


def publish(run_dir: Path, patterns: List[str]) -> List[Path]:
    """
    Copy the files of a training folder (inside RUNS_DIR) that match `patterns` (globs relative to
    run_dir, ex: "model.pt", "yolo/*.png") to the same place in <root>/runs, for webcam.py on Windows.
    The files are copied in the order of the patterns: put the model last, so that webcam.py never
    lists a folder whose model is not complete. Nothing to do if RUNS_DIR is <root>/runs.
    :return: the copied files.
    """
    if not publishing():
        return []
    target_dir = PUBLISH_DIR / run_dir.resolve().relative_to(RUNS_DIR.resolve())
    copied = []
    for pattern in patterns:
        for source in sorted(run_dir.glob(pattern)):
            if source.is_file():
                target = target_dir / source.relative_to(run_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(target)
    print(f"Published for webcam.py: {len(copied)} file(s) in {target_dir}")
    return copied


if __name__ == "__main__":
    print(f"System:       {SYSTEM}")
    print(f"Settings:     {PATHS_FILE} ({'found' if PATHS_FILE.exists() else 'not found, defaults'})")
    print(f"scryfall_dir: {SCRYFALL_DIR}")
    print(f"runs_dir:     {RUNS_DIR}")
    print(f"Models published into {PUBLISH_DIR}: {'yes' if publishing() else 'no (runs_dir is already there)'}")

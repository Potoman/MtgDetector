"""
The folders of the project, both inside the repository (<root>) and ignored by git:
- <root>/scryfall (SCRYFALL_DIR): what download.py gets from scryfall: the cards,
  exp/exp_<code>/info.json and exp/exp_<code>/image/*.jpg (EXP_DIR), the card symbols (symbol/),
  the set symbols (set/, set_png/).
- <root>/runs (RUNS_DIR): what the trainings write (keys, datasets, check images, Ultralytics runs,
  models), where webcam.py finds the models.
"""
from pathlib import Path

# Project root (<root>), independent of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
SCRYFALL_DIR = ROOT / "scryfall"
# The expansions: EXP_DIR/exp_<code>/info.json and EXP_DIR/exp_<code>/image/*.jpg.
EXP_DIR = SCRYFALL_DIR / "exp"
RUNS_DIR = ROOT / "runs"

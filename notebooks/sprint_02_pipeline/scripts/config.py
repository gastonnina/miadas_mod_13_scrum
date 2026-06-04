import os
from pathlib import Path

# Resolve project root: scripts/ -> sprint_02_pipeline/ -> notebooks/ -> project root
_default_root = Path(__file__).resolve().parents[3]

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(_default_root)))
RAW_DIR = Path(os.environ.get("RAW_DIR", str(PROJECT_ROOT / "data" / "raw")))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed")))

# Checkpoint file lives alongside the notebooks, one level above scripts/
_default_state = str(Path(__file__).resolve().parents[1] / "pipeline_state.json")
PIPELINE_STATE_FILE = Path(os.environ.get("PIPELINE_STATE_FILE", _default_state))

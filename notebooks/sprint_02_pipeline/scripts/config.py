import os
from pathlib import Path

def _find_default_root() -> Path:
    """Detecta la raiz tanto en el repo local como dentro del contenedor Docker."""
    candidate = Path(__file__).resolve()

    for parent in [candidate, *candidate.parents]:
        if (parent / "data").exists():
            return parent

    # Fallback defensivo: evita IndexError si la jerarquia cambia en runtime.
    parents = candidate.parents
    return parents[len(parents) - 1]


_default_root = _find_default_root()

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(_default_root)))
RAW_DIR = Path(os.environ.get("RAW_DIR", str(PROJECT_ROOT / "data" / "raw")))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed")))

# Checkpoint file lives alongside the notebooks, one level above scripts/
_default_state = str(Path(__file__).resolve().parents[1] / "pipeline_state.json")
PIPELINE_STATE_FILE = Path(os.environ.get("PIPELINE_STATE_FILE", _default_state))

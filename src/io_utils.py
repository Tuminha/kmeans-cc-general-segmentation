from pathlib import Path

def raw_path(filename: str) -> Path:
    """Return Path to data/raw/filename."""
    # TODO: Path(__file__).resolve().parents[1] / "data" / "raw" / filename
    raise NotImplementedError

def ensure_dirs():
    """Create data/interim and artifacts subdirs if missing."""
    # TODO: Path(...).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError


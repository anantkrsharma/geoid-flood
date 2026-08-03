import gzip
import json
from pathlib import Path

S1_MEAN = (-12.59, -20.26)
S1_STD = (5.26, 5.91)


def get_grids(path: str | Path) -> dict[str, dict]:
    """Load KuroSiwo grid metadata from a gzipped JSON file.

    Args:
        path: path to ``kurosiwo_grids.json.gz``.

    Returns:
        Mapping of grid id to ``{"path", "info", "clz", "clz_name"}``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KuroSiwo grid metadata not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)

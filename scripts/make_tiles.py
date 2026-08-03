"""Rebuild the chip inventory (`data_tiles_*.csv`) from a GEOID-Flood tree.

The dataset ships `data_tiles_s256_st128.csv`, the exact inventory the published models were
trained on: 256x256 chips at stride 128 for train and stride 256 (no overlap) for val/test.
That file is one instantiation of a choice, not a property of the dataset -- the rasters
themselves are 1024x1024. This script regenerates it at any tile size and stride, so a 224 or
512 chip grid, or no tiling at all (`--no-tiling`, one row per 1024x1024 raster), needs no
change to the loader: point `data.init_args.metadata_filename` at the new CSV.

Which 1024x1024 tiles are inventoried comes from `tile_catalog.parquet`, selected by the paper's
two criteria: `--valid-only` (`is_valid == True`, on by default) and `--max-invalid-frac` (keep
`invalid_pixel_frac <= 0.95`). With the defaults, the selection is exactly the tile inventory the
paper reports -- 12,853 tiles for `geoid-flood` (8,938 train / 1,241 val / 2,674 test) and 1,429
for `geoid-flood-heldout`. Pass `--no-valid-only` and `--max-invalid-frac 1.0` to visit every tile
in the catalog. `split` and `event_date` are read from the same table, so a regenerated CSV
inherits the published split assignment unchanged.

Per-chip statistics are computed from the label raster exactly as the original tiler computed
them: `valid_proportion` (fraction of pixels != 255), `positive_proportion` (fraction of pixels
in {1, 2}), and, for `s2l2a` only, `cloud_cover` from the matching `cloudmask` raster.
`min_valid_proportion` in the datamodule filters on `valid_proportion`, so re-tiling at a
different size changes which chips survive that filter -- expect different row counts.

Examples:
    # the paper's tile selection, at the shipped 256/128 chip grid
    python scripts/make_tiles.py --data-root data/geoid-flood

    # every tile in the catalog, no selection criteria
    python scripts/make_tiles.py --data-root data/geoid-flood --no-valid-only --max-invalid-frac 1.0

    # 512x512 chips, stride 256 on train
    python scripts/make_tiles.py --data-root data/geoid-flood --tile-size 512 --stride 256

    # no tiling: one row per 1024x1024 raster per modality
    python scripts/make_tiles.py --data-root data/geoid-flood --no-tiling

    # S1-GRD only, one event, to a chosen path
    python scripts/make_tiles.py --data-root data/geoid-flood --modality s1grd \
        --event EMSR332 --output /tmp/emsr332.csv
"""

import argparse
import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window

try:
    from tqdm import tqdm
except ImportError:  # keeps the script runnable straight from the dataset tree

    def tqdm(iterable, **_):
        return iterable


LOG = logging.getLogger("make_tiles")

LABEL_NODATA = 255
DEFAULT_MODALITIES = ("s2l2a", "s1grd", "s1rtc", "dem")
CSV_FIELDS = [
    "tile_id",
    "event_id",
    "modality",
    "x",
    "y",
    "size",
    "event_date",
    "image_time",
    "cloud_cover",
    "positive_proportion",
    "valid_proportion",
    "acquisition_date",
    "label_id",
    "split",
]

IMAGERY_RE = re.compile(r"^(.+?)_(s2l2a|s1grd|s1rtc)_(pre|post)_([^.]+)\.tif$", re.IGNORECASE)


def _parse_imagery_filename(path: Path) -> dict | None:
    match = IMAGERY_RE.match(path.name)
    if not match:
        return None
    return {
        "chip_prefix": match.group(1),
        "modality": match.group(2).lower(),
        "image_time": match.group(3).lower(),
        "acq_date": match.group(4),
    }


def _read_window(path: Path, x: int, y: int, size: int) -> np.ndarray:
    """First band of one window, matching `geoid_flood.io.read_raster(bands=[1], window=...)`."""
    with rio.open(path) as dataset:
        data = dataset.read([1], window=Window(x, y, size, size), boundless=True, fill_value=255)
    return data[0]


def _tile_positions(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    return [
        (col, row)
        for row in range(0, height - tile_size + 1, stride)
        for col in range(0, width - tile_size + 1, stride)
    ]


def _label_window_stats(label_path: Path, x: int, y: int, size: int) -> tuple[float, float]:
    """(positive_proportion, valid_proportion) over one window of the label raster."""
    arr = _read_window(label_path, x, y, size)
    n = size * size
    if n == 0:
        return 0.0, 0.0
    positive = float(np.sum((arr == 1) | (arr == 2)) / n)
    valid = float(np.sum(arr != LABEL_NODATA) / n)
    return positive, valid


def _cloud_cover(cloudmask_path: Path, x: int, y: int, size: int) -> float:
    arr = _read_window(cloudmask_path, x, y, size)
    valid = arr != 255
    if not np.any(valid):
        return 0.0
    return float(np.sum(arr[valid] > 0) / np.sum(valid))


def _acquisition_iso(acq_date: str) -> str:
    if not acq_date:
        return ""
    try:
        return datetime.strptime(acq_date, "%Y%m%dT%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return acq_date


def _chip_sort_key(prefix: str) -> tuple[int, str]:
    match = re.search(r"-(\d+)$", prefix)
    return (int(match.group(1)) if match else -1, prefix)


def _row_key(row: dict) -> tuple:
    return (row["event_id"], row["tile_id"], row["modality"], int(row["x"]), int(row["y"]), int(row["size"]))


def select_chips(
    catalog: pd.DataFrame,
    events: list[str] | None,
    valid_only: bool = True,
    max_invalid_frac: float = 0.95,
) -> dict[tuple[str, int], dict[str, dict]]:
    """Group the requested catalog rows into {(event_id, aoi): {chip_prefix: {event_date, split}}}.

    ``valid_only`` and ``max_invalid_frac`` are the paper's two tile-selection criteria; with their
    defaults the selection is exactly the tile inventory the paper reports.
    """
    df = catalog
    if valid_only:
        df = df[df["is_valid"].eq(True)]
    if max_invalid_frac is not None and "invalid_pixel_frac" in df.columns:
        df = df[df["invalid_pixel_frac"].le(max_invalid_frac)]
    if events:
        wanted = {e.upper() for e in events}
        keep = df.apply(
            lambda r: str(r["event_id"]).upper() in wanted
            or f"{r['event_id']}-{int(r['tile_id'])}".upper() in wanted,
            axis=1,
        )
        df = df[keep]

    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for _, row in df.iterrows():
        aoi = int(row["tile_id"])
        key = (str(row["event_id"]), aoi)
        prefix = f"{row['event_id']}-{aoi}-{int(row['bbox_id'])}"
        event_date = row["event_time"] if pd.notna(row.get("event_time")) else ""
        split = str(row["split"]).strip() if pd.notna(row.get("split")) else ""
        grouped.setdefault(key, {})[prefix] = {"event_date": str(event_date), "split": split}
    return grouped


def chip_rows(
    data_root: Path,
    event_id: str,
    aoi: int,
    chip_prefix: str,
    meta: dict,
    modalities: tuple[str, ...],
    tile_size: int | None,
    stride: int,
    eval_stride: int | None,
) -> list[dict]:
    """Every inventory row for one 1024x1024 tile, across the requested modalities."""
    event_dir = data_root / f"{event_id}-{aoi}"
    label_path = event_dir / "label" / f"{chip_prefix}_label.tif"
    if not label_path.exists():
        LOG.warning("Missing label for %s, skipping tile", chip_prefix)
        return []

    try:
        with rio.open(label_path) as dataset:
            height, width = dataset.height, dataset.width
    except Exception as exc:  # unreadable raster: report and move on
        LOG.warning("Could not read %s: %s", label_path, exc)
        return []

    if tile_size is None:  # --no-tiling: one window covering the whole raster
        size = min(height, width)
        positions = [(0, 0)]
    else:
        size = tile_size
        train_split = meta["split"] == "train"
        effective_stride = stride if train_split else (eval_stride or tile_size)
        positions = _tile_positions(height, width, size, effective_stride)
    if not positions:
        return []

    stats = {(x, y): _label_window_stats(label_path, x, y, size) for x, y in positions}
    label_id = f"{chip_prefix}_label"
    rows: list[dict] = []

    def emit(tile_id: str, modality: str, image_time: str, acquisition_date: str, clouds: dict | None) -> None:
        for x, y in positions:
            positive, valid = stats[(x, y)]
            rows.append(
                {
                    "tile_id": tile_id,
                    "event_id": event_id,
                    "modality": modality,
                    "x": x,
                    "y": y,
                    "size": size,
                    "event_date": meta["event_date"],
                    "image_time": image_time,
                    "cloud_cover": round(clouds[(x, y)], 4) if clouds else 0.0,
                    "positive_proportion": round(positive, 4),
                    "valid_proportion": round(valid, 4),
                    "acquisition_date": acquisition_date,
                    "label_id": label_id,
                    "split": meta["split"],
                }
            )

    for modality in modalities:
        mod_dir = event_dir / modality
        if not mod_dir.exists():
            continue

        if modality == "dem":
            if (mod_dir / f"{chip_prefix}_dem.tif").exists():
                emit(f"{chip_prefix}_dem", "dem", "", "", None)
            continue

        for path in sorted(mod_dir.glob(f"{chip_prefix}_{modality}_*.tif")):
            info = _parse_imagery_filename(path)
            if not info or info["modality"] != modality or info["chip_prefix"] != chip_prefix:
                continue
            clouds = None
            if modality == "s2l2a":
                cloudmask_path = event_dir / "cloudmask" / path.name.replace("s2l2a", "cloudmask")
                if cloudmask_path.exists():
                    clouds = {(x, y): _cloud_cover(cloudmask_path, x, y, size) for x, y in positions}
            emit(path.stem, modality, info["image_time"], _acquisition_iso(info["acq_date"]), clouds)

    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the data_tiles chip inventory from a GEOID-Flood tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Tree root, e.g. data/geoid-flood")
    parser.add_argument("--catalog", type=Path, default=None, help="Default: <data-root>/tile_catalog.parquet")
    parser.add_argument("--output", type=Path, default=None, help="Default: <data-root>/data_tiles_s{size}_st{stride}.csv")
    parser.add_argument("--tile-size", type=int, default=256, help="Chip size in pixels (default: 256)")
    parser.add_argument("--stride", type=int, default=128, help="Stride for the train split (default: 128)")
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=None,
        help="Stride for val/test (default: tile size, i.e. no overlap -- what the release uses)",
    )
    parser.add_argument(
        "--no-tiling",
        action="store_true",
        help="One row per raster instead of a chip grid; overrides --tile-size and --stride",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        choices=list(DEFAULT_MODALITIES),
        dest="modalities",
        help="Modalities to inventory (default: all four)",
    )
    parser.add_argument(
        "--valid-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only tiles with is_valid == True (default: on; --no-valid-only keeps all)",
    )
    parser.add_argument(
        "--max-invalid-frac",
        type=float,
        default=0.95,
        help="Keep tiles with invalid_pixel_frac <= this (default: 0.95; pass 1.0 to keep all)",
    )
    parser.add_argument("--event", nargs="+", default=None, dest="events", help="Limit to EMSR ids or EMSR-AoI dirs")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Append only rows missing from an existing --output CSV (resume an interrupted run)",
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR (default: INFO)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    data_root: Path = args.data_root
    if not data_root.is_dir():
        raise SystemExit(f"Not a directory: {data_root}")

    catalog_path = args.catalog or data_root / "tile_catalog.parquet"
    if not catalog_path.exists():
        raise SystemExit(f"Missing tile catalog: {catalog_path}")

    tile_size = None if args.no_tiling else args.tile_size
    stride = args.stride or args.tile_size
    if args.output is not None:
        output = args.output
    elif args.no_tiling:
        output = data_root / "data_tiles_full.csv"
    else:
        output = data_root / f"data_tiles_s{args.tile_size}_st{args.stride}.csv"

    catalog = pd.read_parquet(catalog_path)
    grouped = select_chips(catalog, args.events, args.valid_only, args.max_invalid_frac)
    n_chips = sum(len(chips) for chips in grouped.values())
    LOG.info(
        "Selected %d tiles across %d event-AoIs from %s (valid_only=%s, invalid_pixel_frac <= %s)",
        n_chips, len(grouped), catalog_path.name, args.valid_only, args.max_invalid_frac,
    )
    if not grouped:
        raise SystemExit("Nothing selected -- check --event / --valid-only / --max-invalid-frac")

    existing: set[tuple] = set()
    if args.skip_existing and output.exists():
        with output.open("r", newline="", encoding="utf-8") as handle:
            existing = {_row_key(row) for row in csv.DictReader(handle)}
        LOG.info("Resuming: %d rows already in %s", len(existing), output)

    rows: list[dict] = []
    for (event_id, aoi), chips in tqdm(sorted(grouped.items()), desc="Event-AoIs"):
        for chip_prefix in sorted(chips, key=_chip_sort_key):
            new = chip_rows(
                data_root,
                event_id,
                aoi,
                chip_prefix,
                chips[chip_prefix],
                tuple(args.modalities),
                tile_size,
                stride,
                args.eval_stride,
            )
            if existing:
                new = [row for row in new if _row_key(row) not in existing]
            rows.extend(new)

    if not rows:
        LOG.info("No rows to write")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.skip_existing and output.exists()) else "w"
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    LOG.info("Wrote %d rows to %s (%s)", len(rows), output, "appended" if mode == "a" else "created")
    return 0


if __name__ == "__main__":
    sys.exit(main())

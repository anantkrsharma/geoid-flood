from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio import logging
from rasterio.windows import Window

log = logging.getLogger()
log.setLevel(logging.ERROR)


def read_raster(
    path: Path,
    bands: list[int] = None,
    window: tuple[int, int, int, int] = None,
    return_profile: bool = False,
    nodata_value: float = np.nan,
) -> np.ndarray:
    """Read a raster file using rasterio.

    Args:
        path (Path): Path to the raster file.
        bands (list[int], optional): List of bands to read. Defaults to None.
        window (tuple[int, int, int, int], optional): Window to read. Defaults to None.
        return_profile (bool, optional): Whether to return the profile. Defaults to False.

    Returns:
        np.ndarray: Raster data.
    """
    with rio.open(path) as dataset:
        options = {}

        if window is not None:
            fill_value = 0 if bands is None else 255
            options.update(window=Window(*window), boundless=True, fill_value=fill_value)
        if bands is not None:
            data = dataset.read(bands, **options)
        else:
            data = dataset.read(**options)

        np.nan_to_num(data, copy=False, nan=nodata_value)

        if return_profile:
            return data, dataset.profile
        return data


def read_raster_profile(path: Path) -> dict:
    """Read a raster file profile using rasterio.

    Args:
        path (Path): Path to the raster file.

    Returns:
        dict: Raster profile.
    """
    with rio.open(path) as dataset:
        profile = dataset.profile
    return profile


def read_raster_bounds(path: Path) -> dict:
    """Read a raster file bounds using rasterio.

    Args:
        path (Path): Path to the raster file.

    Returns:
        dict: Raster bounds.
    """
    with rio.open(path) as dataset:
        bounds = dataset.bounds
    return bounds

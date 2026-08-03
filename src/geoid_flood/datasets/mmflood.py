import json
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from rasterio.warp import Resampling, reproject
from terratorch.datasets.utils import generate_bands_intervals
from torchgeo.datasets.geo import NonGeoDataset

from geoid_flood.io import read_raster

log = logging.getLogger(__name__)


class MMFloodDataset(NonGeoDataset):
    def __init__(
        self,
        split: str,
        root: Path = Path("data/flood-datasets/mmflood"),
        transform: Callable = None,
        no_data_replace: float = 0.0,
        no_label_replace: int = 255,
        input_sources: list[str] = ["s1grd"],
        output_sources: list[str] = ["s1grd"],
        dataset_bands: list[int | str] | None = None,
        output_bands: list[int | str] | None = None,
        concat_bands: bool = True,
        rgb_indices: list[int] | None = None,
        **kwargs,
    ):
        """
        Initialize the MMFlood dataset for the Terratorch Framework.

        MMFlood ships Sentinel-1 GRD imagery (VV/VH, linear scale) under
        ``<root>/EMSR<code>-<aoi>/s1_raw`` and the corresponding flood masks (values {0, 1})
        under ``<root>/EMSR<code>-<aoi>/mask``. Train/val/test splits are defined in
        ``<root>/activations.json`` keyed by EMSR code (``subset`` field).

        Note: MMFlood tiles are variable-sized, so a fixed-size crop transform is required to
        batch the training split, while validation/testing should run full tiles with batch
        size 1 (and tiled inference).

        Parameters:
        split (str): One of "train", "val", or "test".
        root (Path): Path to the root directory of the MMFlood dataset.
        transform (Callable): An optional albumentations transform applied to image and mask.
        no_data_replace (float): The value to replace no-data pixels with.
        no_label_replace (int): The value to replace no-label pixels with.
        input_sources (list[str]): Input sources to load. Only "s1grd" is supported.
        output_sources (list[str]): Output sources to load. Only "s1grd" is supported.
        dataset_bands (list[int | str] | None): Bands present when reading a tile.
        output_bands (list[int | str] | None): Bands to output (subset of dataset_bands).
        concat_bands (bool): Whether to concatenate the bands into a single tensor.
        rgb_indices (list[int] | None): Indices to use as RGB channels for plotting.
        **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__()
        self.root = Path(root)
        assert self.root.exists(), f"root {self.root} does not exist"
        assert split in ("train", "val", "test"), f"split must be one of train, val, test, got {split}"
        self.split = split
        self.transform = transform
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.input_sources = input_sources
        self.output_sources = output_sources
        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices
        self.concat_bands = concat_bands

        self.dataset_bands = generate_bands_intervals(dataset_bands)
        self.output_bands = generate_bands_intervals(output_bands)

        if self.output_bands and not self.dataset_bands:
            msg = "If output bands provided, dataset_bands must also be provided"
            return Exception(msg)

        if self.output_bands:
            if len(set(self.output_bands) & set(self.dataset_bands)) != len(self.output_bands):
                msg = "Output bands must be a subset of dataset bands"
                raise Exception(msg)

            self.filter_indices = [self.dataset_bands.index(band) for band in self.output_bands]
        else:
            self.filter_indices = None

        # Map each EMSR code to its split (subset) from activations.json.
        activations_file = self.root / "activations.json"
        assert activations_file.exists(), f"activations file {activations_file} does not exist"
        with open(activations_file, "r", encoding="utf-8") as f:
            activations = json.load(f)
        code_to_subset = {code: data.get("subset") for code, data in activations.items()}

        # Collect (image, flood mask, hydro) triples for every s1_raw tile belonging to this split.
        # The binary water label is flood (mask folder, == class 2 in GEOID-Flood) OR permanent
        # water (hydro folder, == class 1). Hydro is optional (not every tile has one).
        self.images = []
        self.labels = []
        self.hydros = []
        for event_dir in sorted(self.root.glob("EMSR*-*")):
            if not event_dir.is_dir():
                continue
            code = event_dir.name.split("-")[0]
            if code_to_subset.get(code) != self.split:
                continue
            s1_dir = event_dir / "s1_raw"
            mask_dir = event_dir / "mask"
            hydro_dir = event_dir / "hydro"
            if not s1_dir.is_dir():
                continue
            for image_path in sorted(s1_dir.glob("*.tif")):
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    log.warning(f"Mask {mask_path} missing for image {image_path}, skipping")
                    continue
                hydro_path = hydro_dir / image_path.name
                self.images.append(image_path)
                self.labels.append(mask_path)
                self.hydros.append(hydro_path if hydro_path.exists() else None)

        log.info(f"Loaded {len(self.images)} {self.split} images")
        log.info(f"Loaded {len(self.labels)} {self.split} masks")

    def __len__(self):
        return len(self.images)

    def _read_s1(self, path: Path):
        image = read_raster(path, nodata_value=self.no_data_replace)
        image = 10 * np.log10(image + np.finfo(np.float32).eps)
        return image.astype(np.float32)

    def _read_hydro(self, hydro_path: Path, ref_profile: dict) -> np.ndarray:
        """Read the permanent water (hydro) raster resampled onto the mask grid.

        Hydro shares the mask's CRS and bounds but uses a slightly different pixel grid, so it
        is reprojected (nearest) onto the mask's transform/shape before combining.
        """
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint8)
        with rasterio.open(hydro_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.nearest,
            )
        return dst

    def __getitem__(self, idx: int):
        concat_image = self._read_s1(self.images[idx])
        if self.filter_indices:
            concat_image = concat_image[self.filter_indices]

        mask_path = self.labels[idx]
        mask, mask_profile = read_raster(mask_path, return_profile=True, nodata_value=self.no_label_replace)
        if mask.ndim == 3:
            mask = mask[0]  # (1, H, W) -> (H, W); squeeze before transform so geometric augs treat it as 2D
        # Binary water = flood (mask == 1) OR permanent water (hydro == 1).
        ignore = mask == self.no_label_replace
        water = mask == 1
        hydro_path = self.hydros[idx]
        if hydro_path is not None:
            hydro = self._read_hydro(hydro_path, mask_profile)
            water = water | (hydro == 1)
        mask = water.astype(np.float32)
        mask[ignore] = self.no_label_replace

        if self.transform:
            concat_image = concat_image.transpose(1, 2, 0)
            data = self.transform(image=concat_image, mask=mask)
            data["mask"] = data["mask"].long().squeeze(0)
        else:
            data = {"image": torch.from_numpy(concat_image), "mask": torch.from_numpy(mask)}
            data["mask"] = data["mask"].long().squeeze(0)

        if not self.concat_bands and len(self.output_sources) > 1:  # FIXME: minor hack for terramind
            concat_image = data.pop("image")
            data["sen1grd"] = concat_image[:2, ...]
            data["sen2l2a"] = concat_image[2:, ...]
        data["filename"] = str(mask_path)
        return data

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None, show_axes: bool | None = False
    ) -> Figure:
        """Plot a sample from the dataset (S1 VV channel + masks)."""
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image_s1 = image.take([0], axis=0)
        image_s1 = np.transpose(image_s1, (1, 2, 0))
        denom = image_s1.max(axis=(0, 1)) - image_s1.min(axis=(0, 1))
        denom[denom == 0] = 1
        image_s1 = (image_s1 - image_s1.min(axis=(0, 1))) / denom
        image_s1 = np.clip(image_s1, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        prediction_mask = None
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image_s1,
            label_mask,
            prediction=prediction_mask,
            suptitle=suptitle,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(
        image_s1: np.ndarray,
        label: np.ndarray,
        prediction=None,
        suptitle=None,
        show_axes=False,
    ):
        images = [image_s1, label, prediction]
        titles = ["Image S1", "Ground Truth Mask", "Predicted Mask"]
        num_images = len([img for img in images if img is not None])

        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        image_count = 0
        while image_count < num_images:
            if images[image_count] is None:
                images.pop(image_count)
                titles.pop(image_count)
                continue

            ax[image_count].axis(axes_visibility)
            ax[image_count].title.set_text(titles[image_count])
            ax[image_count].imshow(images[image_count])
            image_count += 1

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

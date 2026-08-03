import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from pyproj import Transformer
from terratorch.datasets.utils import generate_bands_intervals
from torchgeo.datasets.geo import NonGeoDataset

from geoid_flood.io import read_raster, read_raster_bounds, read_raster_profile

log = logging.getLogger(__name__)


class WorldFloodsDataset(NonGeoDataset):

    def __init__(
        self,
        split: str,
        root: Path = Path("data/flood-datasets/WorldFloodsv2"),
        transform: Callable = None,
        no_data_replace: float = 0.0,
        no_label_replace: float = 255,
        input_sources: list[str] = ["RTC", "S2"],
        output_sources: list[str] = ["RTC", "S2"],
        masked_sources: list[str] = [],
        dataset_bands: list[int | str] | None = None,
        output_bands: list[int | str] | None = None,
        max_cc: float = 0.0,
        max_diff: int = None,
        min_water: float = 0.0,
        apply_erosion: bool = False,
        use_metadata: bool = False,
        rgb_indices: list[int] = None,
        concat_bands: bool = True,
        **kwargs,
    ):
        """
        Initialize the WorldFloods dataset for Terratorch Framework.

        Args:
            split: which split to use, one of "train", "val", "test".
            root: root directory of the WorldFloods dataset.
            transform: optional transformation to apply to the images and masks.
            no_data_replace: value to replace no data values with.
            no_label_replace: value to replace no label values with.
            input_sources: list of sources to use for the input data.
            output_sources: list of sources to use for the output data.
            masked_sources: list of sources to mask.
            dataset_bands: list of bands to load from the dataset.
            output_bands: list of bands to output.
            max_cc: maximum cloud cover/invalid values accepted. Images with a cloud cover higher than this value will be filtered out.
            max_diff: maximum difference between the two dates of the two sources.
            min_water: minimum water proportion.
            apply_erosion: whether to apply erosion to the masks.
            use_metadata: whether to use metadata to filter tiles.
            rgb_indices: list of indices to use as RGB channels.
            concat_bands: whether to concatenate the bands.
        """
        self.root = Path(root)
        assert self.root.exists(), f"root {self.root} does not exist"
        assert split in ("train", "val", "test"), f"split must be one of train, val, test, got {split}"
        self.split = split
        self.transform = transform
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.split_path = self.root / self.split
        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices
        self.use_metadata = use_metadata
        self.concat_bands = concat_bands
        self.masked_sources = masked_sources

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

        self.input_sources = input_sources
        self.output_sources = output_sources
        self.max_diff = max_diff
        self.apply_erosion = apply_erosion

        tiles_metadata_path = self.root / "worldfloods_image_stats.csv"
        self.tiles_info = pd.read_csv(tiles_metadata_path)
        self.tiles_info = self.tiles_info.loc[self.tiles_info["split"] == self.split]

        self.images_root_path = {}
        for source in self.input_sources:
            if source == "RTC":
                self.images_root_path[source] = self.split_path / "S1RTC_224"
            elif source == "S2":
                self.images_root_path[source] = self.split_path / "S2_224"
            else:
                raise NotImplementedError(f"Source {source} not implemented")
        self.labels_root_path = self.split_path / "gt_224"

        self.images = []
        self.labels = []

        for _, row in self.tiles_info.iterrows():
            if min_water and row["water_proportion"] < min_water:
                continue
            if max_cc and row["invalid_proportion"] > max_cc:
                continue
            if self.max_diff:
                # Some tiles have missing s1/s2 dates (NaN); skip them since the
                # date-difference constraint cannot be evaluated.
                if pd.isna(row["s2_date"]) or pd.isna(row["s1_date"]):
                    continue
                s2_date = datetime.fromisoformat(str(row["s2_date"]))
                s1_date = datetime.fromisoformat(str(row["s1_date"]))
                if abs(s1_date - s2_date).days > self.max_diff:
                    continue

            image_paths = {source: [] for source in self.input_sources}
            for source in self.input_sources:
                image_paths[source].append(self.images_root_path[source] / f"{row['name']}.tif")
            self.labels.append(self.labels_root_path / f"{row['name']}.tif")
            self.images.append(image_paths)

        log.info(f"Loaded {len(self.images)} {self.split} images")
        log.info(f"Loaded {len(self.labels)} {self.split} masks")

    def __len__(self):
        return len(self.images)

    def _read_rtc(self, path: list, mask_image: bool = False):
        if mask_image:
            image = np.zeros((2, 224, 224), dtype=np.float32)  # FIXME: should be the same size of read image
        else:
            image = read_raster(path[0], nodata_value=self.no_data_replace)
            image = 10 * np.log10(image + np.finfo(np.float32).eps)
        return image

    def _read_s2(self, path: list, mask_image: bool = False):
        if mask_image:
            image = np.zeros((15, 224, 224), dtype=np.float32)  # FIXME: should be the same size of read image
        else:
            image = read_raster(path[0], nodata_value=self.no_data_replace)
            image = image.astype(np.float32)
        return image

    def _concat(self, *images):
        image = np.vstack(images)  # vv, vh
        return image

    def _get_date(self, name: str) -> torch.Tensor:
        date_str = self.tiles_info.loc[self.tiles_info["name"] == name, "event_date"].values[0]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return torch.tensor([[date.year, date.timetuple().tm_yday - 1]], dtype=torch.float32)

    def _get_coords(self, path: Path) -> torch.Tensor:
        bounds = read_raster_bounds(path)
        profile = read_raster_profile(path)
        crs = str(profile["crs"])
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)
        lat_lon = np.asarray([lat, lon])
        return torch.tensor(lat_lon, dtype=torch.float32)

    def __getitem__(self, idx):

        data = []
        for source in self.output_sources:
            masked = source in self.masked_sources
            if "RTC" in source:
                rtc_image = self._read_rtc(self.images[idx][source], mask_image=masked)
                data.append(rtc_image)
            elif "S2" in source:
                s2_image = self._read_s2(self.images[idx][source], mask_image=masked)
                data.append(s2_image)
            else:
                raise ValueError(f"Unknown resource: {source}")
        concat_image = self._concat(*data)
        if self.filter_indices:
            concat_image = concat_image[self.filter_indices, ...]

        mask_path = self.labels[idx]
        mask = read_raster(mask_path, bands=2, nodata_value=self.no_label_replace)
        mask[mask == 0] = self.no_label_replace  # cloud and frame
        mask[mask == 1] = 0  # terrain
        mask[mask == 2] = 1  # water

        if self.split == "train" and self.apply_erosion:
            mask_binary = mask.copy()
            mask_binary[mask_binary != 0] = 1
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
            mask = np.where(((mask_binary == 1) & (mask_eroded == 0)), self.no_label_replace, mask)

        if self.transform is not None:
            concat_image = concat_image.transpose(1, 2, 0)
            data = self.transform(image=concat_image, mask=mask)
            data["mask"] = data["mask"].long()
        else:
            data = {"image": torch.from_numpy(concat_image), "mask": torch.from_numpy(mask).long()}

        if self.use_metadata:
            location_coords = self._get_coords(mask_path)
            temporal_coords = self._get_date(Path(mask_path).stem)
            data["latlon"] = location_coords
            data["time"] = temporal_coords

        if not self.concat_bands and len(self.output_sources) > 1:  # FIXME: minor hack for terramind
            concat_image = data.pop("image")
            data["sen1grd"] = concat_image[:2, ...]
            data["sen2l2a"] = concat_image[2:, ...]

        data["filename"] = str(mask_path)
        return data

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None, show_axes: bool | None = False
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
            suptitle (str|None): optional string to use as a suptitle
            show_axes (bool|None): whether to show axes or not

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        if "image" in sample:  # FIXME: minor hack for terramind
            image = sample["image"]
        else:
            s1_image = sample["sen1grd"]
            s2_image = sample["sen2l2a"]
            image = self._concat(s1_image, s2_image)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if "S2" in self.output_sources:
            image_s2 = image.take(self.rgb_indices, axis=0)
            image_s2 = np.transpose(image_s2, (1, 2, 0))
            image_s2 = (image_s2 - image_s2.min(axis=(0, 1))) * (1 / image_s2.max(axis=(0, 1)))
            image_s2 = np.clip(image_s2, 0, 1)
        else:
            image_s2 = None

        if "RTC" in self.output_sources or "SHUB_RTC" in self.output_sources:
            index_s1_bands = [index for index, band in enumerate(self.output_bands) if band in ("VV", "VH")]
            if len(index_s1_bands) == 0:
                log.warning("Cannot plot Sentinel-1 image. Bands VV or VH not found in output_bands")
                image_s1 = None
            else:
                index_s1_bands = index_s1_bands[0]
                image_s1 = image.take([index_s1_bands], axis=0)
                image_s1 = np.transpose(image_s1, (1, 2, 0))
                image_s1 = (image_s1 - image_s1.min(axis=(0, 1))) * (1 / image_s1.max(axis=(0, 1)))
                image_s1 = np.clip(image_s1, 0, 1)
        else:
            image_s1 = None

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image_s2,
            image_s1,
            label_mask,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(
        image_s2: np.ndarray | None,
        image_s1: np.ndarray | None,
        label: np.ndarray,
        prediction=None,
        suptitle=None,
        show_axes=False,
    ):
        images = [image_s2, image_s1, label, prediction]
        titles = ["Image S2", "Image S1", "Ground Truth Mask", "Predicted Mask"]
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

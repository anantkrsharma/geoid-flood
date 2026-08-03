import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from terratorch.datasets.utils import generate_bands_intervals
from torchgeo.datasets.geo import NonGeoDataset

from geoid_flood.io import read_raster

log = logging.getLogger(__name__)


class Sen1Floods11Dataset(NonGeoDataset):
    def __init__(
        self,
        split: str,
        root: Path = Path("data/flood-datasets/sen1floods11/v1.1"),
        transform: Callable = None,
        no_data_replace: float = 0.0,
        no_label_replace: int = 255,
        input_sources: list[str] = ["RTC", "S2"],
        output_sources: list[str] = ["RTC", "S2"],
        dataset_bands: list[int | str] | None = None,
        output_bands: list[int | str] | None = None,
        concat_bands: bool = True,
        rgb_indices: list[int] | None = None,
        min_water: float | None = None,
        max_cc: float | None = None,
        **kwargs,
    ):
        """
        Initialize the Sen1Floods11Dataset for Terratorch Framework. The official hand-labeled
        train/val/test splits are selected based on `split` (flood_{train,valid,test}_data.txt);
        an unknown split falls back to flood_all_data.txt (all hand-labeled tiles).

        Parameters:
        split (str): One of "train", "val", or "test".
        root (Path): Path to the root directory of the Sen1Floods11 dataset.
        transform (Callable): An optional torchvision transform to apply to the images and masks.
        no_data_replace (float): The value to replace no-data pixels with.
        no_label_replace (int): The value to replace no-label pixels with.
        input_sources (list[str]): The list of input sources to load. Each source should be one of "S1Hand","S1RTC","S1SHUBRTC", "S2Hand","S2L2AHand".
        output_sources (list[str]): The list of output sources to load. Each source should be one of "S1Hand","S1RTC","S1SHUBRTC", "S2Hand","S2L2AHand".
        dataset_bands (list[int | str] | None): The list of bands to load from the dataset. If None, all bands are loaded.
        output_bands (list[int | str] | None): The list of bands to output. If None, all bands are outputted.
        concat_bands (bool): Whether to concatenate the bands into a single tensor.
        rgb_indices (list[int] | None): The list of indices to use as RGB channel.
        min_water (float | None): The minimum proportion of water pixels in a tile. If None, no filtering is done.
        max_cc (float | None): The maximum proportion of cloudy pixels in a tile. If None, no filtering is done.
        **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.input_sources = input_sources
        self.output_sources = output_sources
        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices
        self.concat_bands = concat_bands
        self.metadata = pd.read_csv(self.root / "tiles_metadata.csv")
        split_files = {
            "train": "flood_train_data.txt",
            "val": "flood_valid_data.txt",
            "test": "flood_test_data.txt",
        }
        fname = split_files.get(self.split, "flood_all_data.txt")
        candidate = self.root / "splits/flood_handlabeled" / fname
        self.list_tiles_path = (
            candidate if candidate.exists() else self.root / "splits/flood_handlabeled/flood_all_data.txt"
        )
        with open(self.list_tiles_path, "r", encoding="utf-8") as file:
            valid_tiles = [line.rstrip("\n") for line in file]

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

        self.hand_labeled_path = self.root / "data" / "flood_events" / "HandLabeled"
        self.images_root_path = {}
        for source in self.input_sources:
            source_path = self.hand_labeled_path / source
            if not source_path.exists():
                raise FileNotFoundError(f"Source path {source_path} does not exist")
            self.images_root_path[source] = source_path

        self.labels_root_path = self.hand_labeled_path / "LabelHand"

        self.images = []
        self.labels = []
        base_source = self.input_sources[0]
        base_source_path = self.images_root_path[base_source]
        for image_path in base_source_path.glob(f"*.tif"):
            image_paths = {}
            tile_name = "_".join(image_path.stem.split("_")[:2])
            if tile_name not in valid_tiles:
                continue
            if max_cc:
                if tile_name not in self.metadata["tile_id"].values:
                    continue
                if self.metadata.loc[self.metadata["tile_id"] == tile_name, "invalid_proportion"].values[0] > max_cc:
                    continue
            if min_water:
                if tile_name not in self.metadata["tile_id"].values:
                    continue
                if self.metadata.loc[self.metadata["tile_id"] == tile_name, "water_proportion"].values[0] < min_water:
                    continue
            for source in self.input_sources:
                image_name = image_path.name.replace(base_source, source)
                source_path = self.images_root_path[source] / image_name
                if source_path.exists():
                    image_paths[source] = source_path
            if len(image_paths) == len(self.input_sources):
                self.images.append(image_paths)
                label_name = image_path.name.replace(base_source, "LabelHand")
                label_path = self.labels_root_path / label_name
                if label_path.exists():
                    self.labels.append(label_path)
                else:
                    raise FileNotFoundError(f"Label path {label_path} does not exist")

        log.info(f"Loaded {len(self.images)} {self.split} images")
        log.info(f"Loaded {len(self.labels)} {self.split} masks")

    def __len__(self):
        return len(self.images)

    def _read_rtc(self, path: list, mask_image: bool = False):
        image = read_raster(path, nodata_value=self.no_data_replace)
        image = 10 * np.log10(image + np.finfo(np.float32).eps)
        return image

    def _read_s2(self, path: list, mask_image: bool = False):
        image = read_raster(path, nodata_value=self.no_data_replace)
        image = image.astype(np.float32)
        return image

    def __getitem__(self, idx: int):
        data = []
        for source in self.output_sources:
            if "S1" in source or "RTC" in source:
                image = self._read_rtc(self.images[idx][source])
                data.append(image)
            elif "S2" in source:
                image = self._read_s2(self.images[idx][source])
                data.append(image)
            else:
                raise ValueError(f"Unknown source {source}")
        concat_image = np.vstack(data)
        if self.filter_indices:
            concat_image = concat_image[self.filter_indices]

        mask_path = self.labels[idx]
        mask = read_raster(mask_path, nodata_value=self.no_label_replace)
        if mask.ndim == 3:
            mask = mask[0]  # (1, H, W) -> (H, W); squeeze before transform so geometric augs (e.g. D4) treat it as 2D
        mask[mask == -1] = self.no_label_replace
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

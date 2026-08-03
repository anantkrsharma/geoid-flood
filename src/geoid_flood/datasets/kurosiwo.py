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
from geoid_flood.utils import get_grids

log = logging.getLogger(__name__)


class KuroSiwoDataset(NonGeoDataset):
    SAR_BAND_ORDER = ("VH", "VV")
    SAR_BAND_ALIASES = {
        "VH": "VH",
        "VV": "VV",
        "IVH": "VH",
        "IVV": "VV",
    }

    ACTIVATIONS = {
        "train": [
            130,
            470,
            205,
            555,
            118,
            174,
            324,
            421,
            554,
            427,
            518,
            502,
            498,
            497,
            496,
            492,
            147,
            267,
            273,
            275,
            417,
            567,
        ],
        "val": [514, 559, 279, 520, 437],
        "test": [321, 561, 445, 562, 411, 277],
    }

    def __init__(
        self,
        split: str,
        root: Path = Path("data/flood-datasets/KuroSiwo"),
        grids: Path = Path("resources/kurosiwo_grids.json.gz"),
        transform: Callable = None,
        no_data_replace: float = 0.0,
        no_label_replace: float = 255,
        times: list[str] = ["SL1", "SL2", "MS1"],
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
        Initialize the KuroSiwoDataset for Terratorch Framework.

        Parameters:
        split (str): One of "train", "val", or "test".
        root (Path): Path to the root directory of the KuroSiwo dataset.
        grids (Path): Path to the gzipped JSON file containing the grid information.
        transform (Callable): An optional torchvision transform to apply to the images and masks.
        no_data_replace (float): The value to replace no-data pixels with.
        no_label_replace (float): The value to replace no-label pixels with.
        times (list[str]): The list of times to load. Each time should be one of "SL1", "SL2", or "MS1".
        input_sources (list[str]): The list of input sources to load. Each source should be one of "S1", "SHUB_RTC", "RTC", "S2".
        output_sources (list[str]): The list of output sources to load. Each source should be one of "S1", "SHUB_RTC", "RTC", "S2".
        masked_sources (list[str]): The list of sources to mask.
        dataset_bands (list[int | str] | None): The list of bands to load from the dataset. If None, all bands are loaded.
        output_bands (list[int | str] | None): The list of bands to output. If None, all bands are outputted.
        max_cc (float): The maximum proportion of cloudy pixels in a tile.
        max_diff (int): The maximum difference in days between the earliest and latest images in a tile.
        min_water (float): The minimum proportion of water pixels in a tile.
        apply_erosion (bool): Whether to apply erosion to the masks.
        use_metadata (bool): Whether to use metadata to filter tiles.
        rgb_indices (list[int] | None): The list of indices to use as RGB channels. If None, the first three bands are used.
        concat_bands (bool): Whether to concatenate the bands.
        **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__()
        self.split = split
        self.root = Path(root)
        self.valid_acts = self.ACTIVATIONS[split]
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.transform = transform
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

        if max_cc or min_water:
            tiles_info_path = self.root / "kurosiwo_multimodal_statistics.csv"
            tiles_info = pd.read_csv(tiles_info_path)
            tiles_info.set_index("s2_path", inplace=True)

        self.resources = []
        for time in times:
            for source in self.input_sources:
                self.resources.append(f"{time}_{source}")
        # label
        label_root_name = "MK0_MLU"
        self.grids = get_grids(grids)

        self.images = []
        self.labels = []

        for id, metadata in self.grids.items():
            if metadata["info"]["actid"] not in self.valid_acts:  # if event is selected
                continue

            metadata_path = self.root / metadata["path"]
            list_resources = list(Path(metadata_path).iterdir())
            label_list = [p for p in list_resources if p.name.startswith(label_root_name)]

            if len(label_list) != 1:
                log.warning(
                    f"Skipping grid {id}: Found {len(label_list)} labels (expected 1) "
                    f"matching '{label_root_name}*.tif' in {metadata_path}"
                )
                continue
            label_path = label_list[0]
            for time in times:
                sources_dict = {}
                for source in self.input_sources:
                    resources_list = self._select_resources(list_resources, time=time, source=source)
                    if resources_list:
                        sources_dict[source] = resources_list
                if len(sources_dict) != len(self.input_sources):
                    continue  # check if all sources are present in sources_dict

                if self.max_diff is not None:
                    try:
                        all_dates = []
                        for source in sources_dict.values():
                            all_dates.append(datetime.strptime(source[0].stem.split("_")[-1], "%Y%m%d"))
                        if (
                            time == "MS1"
                        ):  # label has always the "post-event" date so this check makes sense only for post-event images
                            all_dates.append(datetime.strptime(label_path.stem.split("_")[-1], "%Y%m%d"))
                        all_dates = sorted(all_dates)

                        if abs(all_dates[0] - all_dates[-1]).days > self.max_diff:
                            continue
                    except Exception as e:
                        log.warning(f"Skipping grid {id}, time {time}: {e}")
                        continue

                if max_cc or min_water:
                    # read metadata file
                    try:
                        s2_path_tail = "/".join(str(sources_dict["S2"][0]).split("/")[-4:])
                        tile_info_row = tiles_info.loc[s2_path_tail]
                    except KeyError:
                        continue
                if max_cc:
                    tile_cc = tile_info_row["invalid_proportion"]
                    if tile_cc > max_cc:
                        continue
                if min_water:
                    tile_water = tile_info_row["water_proportion"]
                    if tile_water < min_water:
                        continue

                self.images.append(sources_dict)
                self.labels.append(label_path)

        log.info(f"Loaded {len(self.images)} {self.split} images from KuroSiwo")
        log.info(f"Loaded {len(self.labels)} {self.split} masks from KuroSiwo")

    def __len__(self):
        return len(self.images)

    def _canonical_sar_band(self, band_token: str) -> str | None:
        return self.SAR_BAND_ALIASES.get(band_token.upper())

    def _extract_sar_band(self, path: Path) -> str | None:
        for token in path.stem.split("_"):
            canonical = self._canonical_sar_band(token)
            if canonical:
                return canonical
        return None

    def _sort_sar_band_paths(self, resources_list: list[Path]) -> list[Path]:
        by_band = {}
        for path in resources_list:
            band = self._extract_sar_band(path)
            if band and band not in by_band:
                by_band[band] = path

        if all(band in by_band for band in self.SAR_BAND_ORDER):
            return [by_band[band] for band in self.SAR_BAND_ORDER]
        return []

    def _select_sourceless_s1_resources(self, list_resources: list[Path], time: str) -> list[Path]:
        prefix = f"{time}_"
        source_less_paths = []
        for path in list_resources:
            if not path.name.startswith(prefix):
                continue
            # Source-less S1 expects second token to be a SAR band (e.g. IVH, IVV, VH, VV).
            stem_tokens = path.stem.split("_")
            if len(stem_tokens) < 2:
                continue
            if self._canonical_sar_band(stem_tokens[1]) is None:
                continue
            source_less_paths.append(path)
        return self._sort_sar_band_paths(source_less_paths)

    def _select_resources(self, list_resources: list[Path], time: str, source: str) -> list[Path]:
        if source == "S1":
            return self._select_sourceless_s1_resources(list_resources, time=time)

        resources_list = [path for path in list_resources if path.name.startswith(f"{time}_{source}_")]
        if "RTC" in source:
            return self._sort_sar_band_paths(resources_list)
        return resources_list

    def _read_rtc(self, path: list, mask_image: bool = False):
        vh_path, vv_path = path
        if mask_image:
            image = np.zeros((2, 224, 224), dtype=np.float32)
        else:
            vh_image, vv_image = read_raster(vh_path, nodata_value=self.no_data_replace), read_raster(
                vv_path, nodata_value=self.no_data_replace
            )
            image = self._concat(vv_image, vh_image)
            image = 10 * np.log10(image + np.finfo(np.float32).eps)
        return image, [str(vv_path), str(vh_path)]

    def _read_s2(self, path: list, mask_image: bool = False):
        if mask_image:
            image = np.zeros((14, 224, 224), dtype=np.float32)
        else:
            image = read_raster(path[0], nodata_value=self.no_data_replace)
        return image, str(path[0])

    def _concat(self, *images):
        image = np.vstack(images)  # vv, vh
        return image

    def _get_date(self, path: Path) -> torch.Tensor:
        date = datetime.strptime(path.stem.split("_")[-1], "%Y%m%d")
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
        image_paths = []
        for source in self.output_sources:
            masked = source in self.masked_sources
            if "RTC" in source or "S1" in source:
                rtc_image, rtc_paths = self._read_rtc(self.images[idx][source], mask_image=masked)
                data.append(rtc_image)
                image_paths.extend(rtc_paths)
            elif "S2" in source:
                s2_image, s2_path = self._read_s2(self.images[idx][source], mask_image=masked)
                data.append(s2_image)
                image_paths.append(s2_path)
            else:
                raise ValueError(f"Unknown resource: {source}")

        concat_image = self._concat(*data)

        if self.filter_indices:
            concat_image = concat_image[self.filter_indices, ...]

        mask_path = self.labels[idx]
        mask = read_raster(mask_path, nodata_value=self.no_label_replace).squeeze(0)
        mask[mask == 3] = self.no_label_replace

        if "S2" in self.output_sources:
            cloud_mask = Path(s2_path).parent / "clouds_l2a" / Path(s2_path).name
            if cloud_mask.exists():
                cloud_mask = read_raster(cloud_mask, nodata_value=1).squeeze(0)
                mask[cloud_mask > 0] = self.no_label_replace

        # Pre & Post label handling (pre only permanent, post permanent + temporary)
        if Path(image_paths[0]).name.startswith("MS1"):
            mask[mask == 2] = 1
        else:
            mask[mask == 2] = 0

        if self.split == "train" and self.apply_erosion:  # FIXME: Add option
            mask_binary = mask.copy()
            mask_binary[mask_binary != 0] = 1
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
            mask = np.where(((mask_binary == 1) & (mask_eroded == 0)), self.no_label_replace, mask)

        if self.transform is not None:
            concat_image = concat_image.transpose(1, 2, 0)  # albumentations expects HWC
            data = self.transform(image=concat_image, mask=mask)
            data["mask"] = data["mask"].long()
        else:
            data = {"image": torch.from_numpy(concat_image), "mask": torch.from_numpy(mask).long()}

        if self.use_metadata:
            location_coords = self._get_coords(image_paths[0])
            temporal_coords = self._get_date(Path(image_paths[0]))
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

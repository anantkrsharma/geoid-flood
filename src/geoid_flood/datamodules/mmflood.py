import logging

import albumentations as A
import kornia.augmentation as K
import torch
from terratorch.datamodules.utils import wrap_in_compose_is_list
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from geoid_flood.datasets import MMFloodDataset

LOG = logging.getLogger(__name__)

# MMFlood-specific Sentinel-1 GRD statistics (dB), from the legacy MMFloodDataset.
MEANS = {
    "AOT": 0.0,
    "COASTAL_AEROSOL": 793.243,
    "BLUE": 1924.863,
    "GREEN": 2184.553,
    "RED": 2340.936,
    "RED_EDGE_1": 2671.402,
    "RED_EDGE_2": 3240.082,
    "RED_EDGE_3": 3468.412,
    "NIR_BROAD": 3563.244,
    "NIR_NARROW": 3627.704,
    "WATER_VAPOR": 3711.071,
    "SWIR_1": 3416.714,
    "SWIR_2": 2849.625,
    "WVP": 0.0,
    "VV": -12.59,
    "VH": -20.26,
}

STDS = {
    "AOT": 1.0,
    "COASTAL_AEROSOL": 1160.144,
    "BLUE": 1201.092,
    "GREEN": 1219.943,
    "RED": 1397.225,
    "RED_EDGE_1": 1400.035,
    "RED_EDGE_2": 1373.136,
    "RED_EDGE_3": 1429.17,
    "NIR_BROAD": 1485.025,
    "NIR_NARROW": 1447.836,
    "WATER_VAPOR": 1652.703,
    "SWIR_1": 1471.002,
    "SWIR_2": 1365.307,
    "WVP": 1.0,
    "VV": 5.26,
    "VH": 5.91,
}


class MMFloodDataModule(NonGeoDataModule):

    def __init__(
        self,
        root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        predict_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        no_data_replace: float | None = 0.0,
        no_label_replace: float | None = 255,
        input_sources: tuple = ("s1grd",),
        output_sources: tuple = ("s1grd",),
        dataset_bands: list[str] | None = None,
        output_bands: list[int | str] | None = None,
        concat_bands: bool = True,
        rgb_indices: list[int] | None = None,
        means: dict[str, float] | None = None,
        stds: dict[str, float] | None = None,
        **kwargs,
    ):
        """
        Initialize the DataModule for the MMFlood dataset for the TerraTorch Framework.

        MMFlood S1 GRD tiles are variable-sized. The training split therefore relies on a
        fixed-size crop in ``train_transform`` so samples can be batched, while
        validation/test/predict run one full tile at a time (batch size forced to 1) and
        should be paired with tiled inference in the model config.

        Args:
            root: root directory of the dataset.
            batch_size: training batch size (val/test/predict are forced to 1).
            num_workers: number of workers to use for data loading.
            train_transform: transformation to apply to the training data (must crop to a
                fixed size, e.g. RandomCrop, so the batch can be collated).
            val_transform: transformation to apply to the validation data.
            test_transform: transformation to apply to the test data.
            predict_transform: transformation to apply to the prediction data.
            drop_last: Whether to drop the last (incomplete) training batch.
            no_data_replace: value to replace no data values with.
            no_label_replace: value to replace no label values with.
            input_sources: data sources of the input data (only "s1grd").
            output_sources: data sources of the output data (only "s1grd").
            dataset_bands: expected bands when reading a tile.
            output_bands: outputted bands by the dataset class.
            concat_bands: whether to concatenate the bands into a single tensor.
            rgb_indices: indices of the RGB bands.
            means: dictionary mapping band names to mean values for normalization. If None, uses default MEANS.
            stds: dictionary mapping band names to standard deviation values for normalization. If None, uses default STDS.
        """
        super().__init__(MMFloodDataset, batch_size, num_workers, **kwargs)

        self.root = root

        # Use provided means/stds or fallback to defaults
        means_dict = means if means is not None else MEANS
        stds_dict = stds if stds is not None else STDS

        LOG.info(f"Using means: {means_dict}")
        LOG.info(f"Using stds: {stds_dict}")

        # Validate that all output_bands are present in means/stds dictionaries
        if output_bands is not None:
            missing_bands = [band for band in output_bands if band not in means_dict]
            if missing_bands:
                raise ValueError(f"Missing bands in means dictionary: {missing_bands}")
            missing_bands = [band for band in output_bands if band not in stds_dict]
            if missing_bands:
                raise ValueError(f"Missing bands in stds dictionary: {missing_bands}")

        # Build normalization arrays
        norm_means = []
        norm_stds = []
        if output_bands is not None:
            for band in output_bands:
                norm_means.append(means_dict[band])
                norm_stds.append(stds_dict[band])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.aug = AugmentationSequential(K.Normalize(norm_means, norm_stds), data_keys=["image"])
        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        # MMFlood tiles are variable-sized: evaluate one full tile at a time.
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.predict_batch_size = 1

        self.input_sources = input_sources
        self.output_sources = output_sources
        self.concat_bands = concat_bands
        self.rgb_indices = rgb_indices
        self.dataset_bands = dataset_bands
        self.output_bands = output_bands

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                root=self.root,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                input_sources=self.input_sources,
                output_sources=self.output_sources,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                concat_bands=self.concat_bands,
                rgb_indices=self.rgb_indices,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                root=self.root,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                input_sources=self.input_sources,
                output_sources=self.output_sources,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                concat_bands=self.concat_bands,
                rgb_indices=self.rgb_indices,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                root=self.root,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                input_sources=self.input_sources,
                output_sources=self.output_sources,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                concat_bands=self.concat_bands,
                rgb_indices=self.rgb_indices,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                root=self.root,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                input_sources=self.input_sources,
                output_sources=self.output_sources,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                concat_bands=self.concat_bands,
                rgb_indices=self.rgb_indices,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=True,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Apply batch-level augmentations after moving to device."""
        if isinstance(batch, dict) and "image" in batch:
            image_tensor = batch["image"]
            if isinstance(image_tensor, torch.Tensor):
                batch["image"] = self.aug(image_tensor)
            else:
                raise ValueError(f"Expected image tensor to be of type torch.Tensor, but got {type(image_tensor)}")

        return batch

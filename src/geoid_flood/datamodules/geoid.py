import albumentations as A
import kornia.augmentation as K
import torch
from terratorch.datamodules.utils import wrap_in_compose_is_list
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule

from geoid_flood.datasets.geoid import (
    BACKBONE_MODALITY_KEYS,
    GEOIDFloodDataset,
    MODALITY_OUTPUT_CHANNELS,
)

MEANS = {
    "VV": -10.089,
    "VH": -17.288,
    "DEM": 670.665,
}

STDS = {
    "VV": 4.770,
    "VH": 7.111,
    "DEM": 951.272,
}


class GEOIDFloodDataModule(NonGeoDataModule):
    """DataModule for the GEOID-Flood dataset with Terratorch/TorchGeo."""

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
        no_data_replace: float = 0.0,
        no_label_replace: int = 255,
        modalities: list[str] | None = None,
        modalities_pre: list[str] | None = None,
        modalities_post: list[str] | None = None,
        image_scope: list[str] | None = None,
        output_bands: list[str] | None = None,
        metadata_filename: str = "data_tiles_s256_st128.csv",
        max_cloud_cover: float = 1.0,
        min_positive: float = 0.0,
        min_positive_train: float | None = None,
        min_valid_proportion: float = 0.0,
        means: dict[str, float] | None = None,
        stds: dict[str, float] | None = None,
        predict_split: str = "all",
        return_timestamps: bool = False,
        return_image_as_dict: bool = False,
        paired: bool = False,
        fuse_paired: bool = False,
        skip_bad_samples: bool = True,
        max_skip_retries: int = 16,
        **kwargs,
    ):
        """Initialize GEOIDFloodDataModule.

        Args:
            root: Root directory of data/geoid-flood.
            batch_size: Batch size for training.
            num_workers: Number of data loader workers.
            train_transform: Albumentations transform for training.
            val_transform: Albumentations transform for validation.
            test_transform: Albumentations transform for test.
            predict_transform: Albumentations transform for prediction.
            drop_last: Whether to drop last incomplete batch.
            no_data_replace: Value to replace no-data pixels.
            no_label_replace: Ignore index for mask.
            modalities: Modalities for both pre/post when modalities_pre/post are unset.
            modalities_pre: Modalities for pre images in paired mode (e.g. ["s2l2a"]).
            modalities_post: Modalities for post images in paired mode (e.g. ["s1grd"]).
            image_scope: Temporal scope (["pre"], ["post"], or ["pre", "post"]).
            output_bands: Bands to output (e.g. ["VV", "VH"]).
            metadata_filename: CSV filename in data root.
            max_cloud_cover: Maximum cloud cover filter.
            min_positive: Minimum positive proportion filter (val/test).
            min_positive_train: Override min_positive for train only (default: use min_positive).
            min_valid_proportion: Minimum valid proportion filter.
            means: Band means for normalization.
            stds: Band stds for normalization.
            predict_split: Split used for predict dataloader ("all", "test", "val", or "train"). Default "all".
            return_timestamps: If True, add "timestamps" to each sample (e.g. for OlmoEarth). Default False.
            return_image_as_dict: If True, dataset returns "image" as dict {modality: tensor} (e.g. for TerraMind).
                When False, image is a single concatenated tensor. Default False.
            fuse_paired: If True (requires paired=True), early-fusion mode: pre and post images are concatenated
                along the channel dimension into a single "image", and the raw three-class mask (0=background,
                1=permanent water, 2=flood) is returned instead of binary remapped masks. Incompatible with
                return_image_as_dict. Default False.
        """
        super().__init__(GEOIDFloodDataset, batch_size, num_workers, **kwargs)

        self.root = root
        means_dict = means if means is not None else MEANS
        stds_dict = stds if stds is not None else STDS

        output_bands = output_bands or ["VV", "VH"]
        missing = [b for b in output_bands if b not in means_dict]
        if missing:
            raise ValueError(f"Missing bands in means: {missing}")
        missing = [b for b in output_bands if b not in stds_dict]
        if missing:
            raise ValueError(f"Missing bands in stds: {missing}")

        norm_means = [means_dict[b] for b in output_bands]
        norm_stds = [stds_dict[b] for b in output_bands]

        self.return_image_as_dict = return_image_as_dict
        self.paired = paired
        self.fuse_paired = fuse_paired
        if fuse_paired and not paired:
            raise ValueError("fuse_paired=True requires paired=True")
        if fuse_paired and return_image_as_dict:
            raise ValueError("fuse_paired=True is incompatible with return_image_as_dict=True")
        self._norm_per_modality = None
        base_modalities = modalities or ["s1grd"]
        if modalities_pre is not None or modalities_post is not None:
            self.modalities_pre = list(modalities_pre if modalities_pre is not None else base_modalities)
            self.modalities_post = list(modalities_post if modalities_post is not None else base_modalities)
            self.all_modalities = list(dict.fromkeys(self.modalities_pre + self.modalities_post))
            self.modalities = list(modalities) if modalities is not None else self.all_modalities
        else:
            self.modalities = list(base_modalities)
            self.modalities_pre = list(self.modalities)
            self.modalities_post = list(self.modalities)
            self.all_modalities = list(self.modalities)
        if return_image_as_dict:
            # Build per-modality normalization (keys match dataset's BACKBONE_MODALITY_KEYS)
            offset = 0
            self._norm_per_modality = {}
            for mod in self.all_modalities:
                nch = MODALITY_OUTPUT_CHANNELS.get(mod, 1)
                bands = output_bands[offset : offset + nch]
                offset += nch
                key = BACKBONE_MODALITY_KEYS.get(mod, mod)
                self._norm_per_modality[key] = (
                    [means_dict[b] for b in bands],
                    [stds_dict[b] for b in bands],
                )

        if fuse_paired:
            # Early fusion concatenates pre then post channels into one image tensor, so the
            # normalization vector must cover both scopes (e.g. [VV, VH, VV, VH] for symmetric S1).
            mod_bands_map: dict[str, list[str]] = {}
            offset = 0
            for mod in self.all_modalities:
                nch = MODALITY_OUTPUT_CHANNELS.get(mod, 1)
                mod_bands_map[mod] = output_bands[offset : offset + nch]
                offset += nch
            fused_bands: list[str] = []
            for mod in self.modalities_pre + self.modalities_post:
                fused_bands += mod_bands_map[mod]
            norm_means = [means_dict[b] for b in fused_bands]
            norm_stds = [stds_dict[b] for b in fused_bands]

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.aug = K.AugmentationSequential(K.Normalize(norm_means, norm_stds), data_keys=["image"])
        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.image_scope = image_scope if image_scope is not None else ["pre", "post"]
        self.output_bands = output_bands
        self.metadata_filename = metadata_filename
        self.max_cloud_cover = max_cloud_cover
        self.min_positive = min_positive
        self.min_positive_train = min_positive_train if min_positive_train is not None else min_positive
        self.min_valid_proportion = min_valid_proportion
        self.predict_split = predict_split
        self.return_timestamps = return_timestamps
        self.skip_bad_samples = skip_bad_samples
        self.max_skip_retries = max_skip_retries

    def setup(self, stage: str) -> None:
        """Set up datasets."""

        def _base_kw(split: str) -> dict:
            return dict(
                data_root=self.root,
                modalities=self.modalities,
                modalities_pre=self.modalities_pre,
                modalities_post=self.modalities_post,
                image_scope=self.image_scope,
                metadata_filename=self.metadata_filename,
                max_cloud_cover=self.max_cloud_cover,
                min_positive=self.min_positive_train if split == "train" else self.min_positive,
                min_valid_proportion=self.min_valid_proportion,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                output_bands=self.output_bands,
                return_timestamps=self.return_timestamps,
                return_image_as_dict=self.return_image_as_dict,
                paired=self.paired,
                fuse_paired=self.fuse_paired,
                skip_bad_samples=self.skip_bad_samples,
                max_skip_retries=self.max_skip_retries,
            )

        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                transform=self.train_transform,
                **_base_kw("train"),
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                transform=self.val_transform,
                **_base_kw("val"),
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                transform=self.test_transform,
                **_base_kw("test"),
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split=self.predict_split,
                transform=self.predict_transform,
                **_base_kw("test"),
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        """Create DataLoader for the given split."""
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
        """Apply batch normalization after transfer to device."""
        if not isinstance(batch, dict):
            return batch

        def _normalize_image(image_tensor):
            if isinstance(image_tensor, dict):
                for key, tensor in image_tensor.items():
                    means, stds = self._norm_per_modality[key]
                    image_tensor[key] = K.Normalize(means, stds)(tensor)
                return image_tensor
            if isinstance(image_tensor, torch.Tensor):
                return self.aug(image_tensor)
            raise ValueError(f"Expected image tensor or dict of tensors, got {type(image_tensor)}")

        if "image" in batch:
            batch["image"] = _normalize_image(batch["image"])
        if "image_pre" in batch:
            batch["image_pre"] = _normalize_image(batch["image_pre"])
        if "image_post" in batch:
            batch["image_post"] = _normalize_image(batch["image_post"])
        return batch

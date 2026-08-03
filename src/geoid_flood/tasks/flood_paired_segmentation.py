"""Paired pre/post segmentation task with additional flood-change loss.

Uses a shared encoder-decoder model applied separately to pre and post images.
Primary supervision is standard CE loss on pre/post masks; an extra BCE term
penalizes disagreement between pre and post logits on newly flooded pixels.
"""

from typing import Any

import torch
from torch import Tensor, nn

from terratorch.models.model import ModelOutput
from terratorch.tasks.segmentation_tasks import (
    SemanticSegmentationTask,
    to_segmentation_prediction,
)


class FloodPairedSegmentationTask(SemanticSegmentationTask):
    """Semantic segmentation task for paired pre/post inputs with flood-change loss.

    Expected batch keys from the datamodule:
    - ``image_pre``, ``image_post``: input tensors (B, C, H, W) or modality dicts
    - ``mask_pre``, ``mask_post``: remapped labels (0=background, 1=water, ignore_index)
    - optional ``timestamps_pre`` when timestamps are used
    """

    def __init__(self, *args: Any, flood_loss_weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.flood_loss_weight = float(flood_loss_weight)
        self._bce = nn.BCEWithLogitsLoss(reduction="mean")

    def _forward_single(self, x: Any, extra: dict[str, Any]) -> ModelOutput:
        """Forward helper that mirrors SemanticSegmentationTask.__call__ usage."""
        return self(x, **extra)

    def _compute_flood_change_loss(
        self,
        logits_pre: Tensor,
        logits_post: Tensor,
        mask_pre: Tensor,
        mask_post: Tensor,
    ) -> Tensor:
        """Compute BCE loss on flood-change signal from pre/post logits and masks.

        Flood target is approximated as pixels that are water in post but not in pre:
        target = 1 where (mask_post == 1 and mask_pre == 0), else 0, ignoring pixels
        with ignore_index in either mask.
        """
        if self.flood_loss_weight <= 0:
            return torch.zeros((), device=logits_pre.device, dtype=logits_pre.dtype)

        assert logits_pre.shape == logits_post.shape, "Pre/post logits must have same shape"
        # Assume class index 1 is water
        water_pre = logits_pre[:, 1]
        water_post = logits_post[:, 1]
        flood_logits = water_post - water_pre  # (B, H, W)

        ignore_index = getattr(self.criterion, "ignore_index", None)
        if ignore_index is None:
            valid = torch.ones_like(mask_pre, dtype=torch.bool)
        else:
            valid = (mask_pre != ignore_index) & (mask_post != ignore_index)

        flood_target = (mask_post == 1) & (mask_pre == 0)
        flood_target = flood_target & valid

        if not torch.any(valid):
            return torch.zeros((), device=logits_pre.device, dtype=logits_pre.dtype)

        flood_logits_flat = flood_logits[valid]
        target_flat = flood_target[valid].to(dtype=flood_logits_flat.dtype)
        return self._bce(flood_logits_flat, target_flat)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute training loss for paired pre/post inputs."""
        x_pre = batch["image_pre"]
        x_post = batch["image_post"]
        y_pre = self.squeeze_ground_truth(batch["mask_pre"])
        y_post = self.squeeze_ground_truth(batch["mask_post"])

        # Only pass model-relevant extras (timestamps for backbones that use them).
        extra_pre: dict[str, Any] = {}
        extra_post: dict[str, Any] = {}

        if "timestamps_pre" in batch:
            extra_pre["timestamps"] = batch["timestamps_pre"]
        if "timestamps_post" in batch:
            extra_post["timestamps"] = batch["timestamps_post"]
        elif "timestamps_pre" in batch:
            extra_post["timestamps"] = batch["timestamps_pre"]

        model_out_pre: ModelOutput = self._forward_single(x_pre, extra_pre)
        model_out_post: ModelOutput = self._forward_single(x_post, extra_post)

        # Main CE losses for pre/post
        loss_dict_pre = self.train_loss_handler.compute_loss(model_out_pre, y_pre, self.criterion, self.aux_loss)
        loss_dict_post = self.train_loss_handler.compute_loss(model_out_post, y_post, self.criterion, self.aux_loss)
        loss_pre = loss_dict_pre["loss"]
        loss_post = loss_dict_post["loss"]

        # Main paired segmentation loss (mean of pre/post CE losses).
        loss = (loss_pre + loss_post) / 2.0

        # Flood-change loss
        flood_loss = self._compute_flood_change_loss(model_out_pre.output, model_out_post.output, y_pre, y_post)
        total_loss = loss + self.flood_loss_weight * flood_loss

        # Logging and metrics based on post prediction
        self.train_loss_handler.log_loss(
            self.log,
            loss_dict={
                "loss": total_loss,
                "loss_pre": loss_pre.detach(),
                "loss_post": loss_post.detach(),
                "loss_flood": flood_loss.detach(),
            },
            batch_size=y_post.shape[0],
        )
        y_hat_post = to_segmentation_prediction(model_out_post)
        self.train_metrics.update(y_hat_post, y_post)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor | None:
        """Validation on post images only, reusing base-class behavior."""
        single_batch = dict(batch)
        single_batch["image"] = single_batch.pop("image_post")
        single_batch["mask"] = single_batch.pop("mask_post")
        single_batch.pop("image_pre", None)
        single_batch.pop("mask_pre", None)
        single_batch.pop("chip_id", None)
        single_batch.pop("split", None)
        single_batch.pop("timestamps_pre", None)
        if "timestamps_post" in single_batch:
            single_batch["timestamps"] = single_batch.pop("timestamps_post")
        return super().validation_step(single_batch, batch_idx, dataloader_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor | None:
        """Test on post images only, reusing base-class behavior."""
        single_batch = dict(batch)
        single_batch["image"] = single_batch.pop("image_post")
        single_batch["mask"] = single_batch.pop("mask_post")
        single_batch.pop("image_pre", None)
        single_batch.pop("mask_pre", None)
        single_batch.pop("chip_id", None)
        single_batch.pop("split", None)
        single_batch.pop("timestamps_pre", None)
        if "timestamps_post" in single_batch:
            single_batch["timestamps"] = single_batch.pop("timestamps_post")
        return super().test_step(single_batch, batch_idx, dataloader_idx)

#!/usr/bin/env python3
"""Paired EMSR eval: binary (post water) and 3-class (no water / permanent water / flood) metrics → TensorBoard.

**Binary** (post only, same as standard 2-class test): target ``mask_post`` (0/1), prediction
``argmax(logits_post)``.

**3-class** (post-aligned, from paired remapped masks): 0 = no water, 1 = water in pre and post
(permanent), 2 = water in post but not in pre (flood / gain). Predictions use the same rule from
``pred_water_pre`` and ``pred_water_post`` (from argmax on pre/post logits).

Metrics match Terratorch ``SemanticSegmentationTask`` style: ``MulticlassJaccardIndex`` and
``MulticlassF1Score`` with the task ``ignore_index`` and ``average="macro"`` where applicable.
Per-class IoU for the 3-class task is logged as ``test/IoU_0``, ``test/IoU_1``, ``test/IoU_2``.

TensorBoard: writing at the same ``global_step`` as the checkpoint replaces prior scalars at that
step in the UI when a single new event is emitted per tag (standard SummaryWriter append behavior).

Requires ``--tensorboard-logdir`` or TensorBoardLogger paths in config.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import torch
import yaml
from terratorch.datamodules.utils import wrap_in_compose_is_list
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

IGNORE_INDEX_DEFAULT = 255


def _resolve_transform(spec: Any) -> Any:
    if isinstance(spec, A.BasicTransform):
        return spec
    if isinstance(spec, dict) and "class_path" in spec:
        class_path = spec["class_path"]
        init_args = spec.get("init_args") or {}
        module_path, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(**init_args)
    raise TypeError(f"Transform spec must be A.BasicTransform or dict with class_path, got {type(spec)}")


def _resolve_transform_list(transform_list: Any) -> Any:
    if transform_list is None:
        return None
    if isinstance(transform_list, (list, tuple)):
        resolved = [_resolve_transform(t) for t in transform_list]
        return wrap_in_compose_is_list(resolved)
    if isinstance(transform_list, dict):
        return _resolve_transform(transform_list)
    return transform_list


def _get_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _ignore_index_from_task(task: Any) -> int:
    crit = getattr(task, "criterion", None)
    if crit is not None and hasattr(crit, "ignore_index"):
        idx = crit.ignore_index
        if idx is not None:
            return int(idx)
    return IGNORE_INDEX_DEFAULT


def _move_tensors_to_device(obj: Any, device: str | torch.device) -> Any:
    """Recursively move tensors to device (handles return_image_as_dict modality dicts)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    return obj


def _trainer_precision(config: dict[str, Any]) -> str | None:
    trainer = config.get("trainer") or {}
    return trainer.get("precision")


def _use_bf16_autocast(precision: str | None, device: str | torch.device) -> bool:
    if not (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()):
        return False
    return precision in ("bf16-mixed", "16-mixed", "16-true")


def _prepare_batch_after_transfer(
    batch: dict[str, Any],
    dm: Any,
    device: str | torch.device,
) -> dict[str, Any]:
    """Match Lightning: transfer to device, then on_after_batch_transfer (normalize on GPU)."""
    batch = _move_tensors_to_device(batch, device)
    return dm.on_after_batch_transfer(batch, 0)


def _task_forward_logits(task: Any, image: Any, extras: dict[str, Any], use_autocast: bool) -> torch.Tensor:
    if use_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return _to_logits(task(image, **extras))
    return _to_logits(task(image, **extras))


def _extras_for_forward_pre(batch: dict[str, Any]) -> dict[str, Any]:
    """Match FloodPairedSegmentationTask.training_step pre forward kwargs."""
    extra: dict[str, Any] = {}
    if "timestamps_pre" in batch:
        extra["timestamps"] = batch["timestamps_pre"]
    return extra


def _extras_for_forward_post(batch: dict[str, Any]) -> dict[str, Any]:
    """Match FloodPairedSegmentationTask.training_step post forward kwargs."""
    extra: dict[str, Any] = {}
    if "timestamps_post" in batch:
        extra["timestamps"] = batch["timestamps_post"]
    elif "timestamps_pre" in batch:
        extra["timestamps"] = batch["timestamps_pre"]
    return extra


def _to_logits(out: Any) -> torch.Tensor:
    if hasattr(out, "output"):
        return out.output
    return out


def _pred_water_bool(logits: torch.Tensor) -> torch.Tensor:
    """(B, H, W) bool: class 1 (water)."""
    return logits.argmax(dim=1) == 1


def _y_binary_post(mask_post: torch.Tensor, valid: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return torch.where(valid, mask_post.long(), torch.full_like(mask_post, ignore_index, dtype=torch.long))


def _pred_binary_post(logits_post: torch.Tensor, valid: torch.Tensor, ignore_index: int) -> torch.Tensor:
    p = logits_post.argmax(dim=1).long()
    return torch.where(valid, p, torch.full_like(p, ignore_index, dtype=torch.long))


def _y_three_class(
    mask_pre: torch.Tensor,
    mask_post: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """0 dry, 1 permanent water, 2 flood (post water, not pre water)."""
    y = torch.full_like(mask_post, ignore_index, dtype=torch.long)
    y[valid & (mask_post == 0)] = 0
    y[valid & (mask_post == 1) & (mask_pre == 1)] = 1
    y[valid & (mask_post == 1) & (mask_pre == 0)] = 2
    return y


def _pred_three_class(
    pred_water_pre: torch.Tensor,
    pred_water_post: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    ref = pred_water_post.long()
    p = torch.full_like(ref, ignore_index, dtype=torch.long)
    p[valid & ~pred_water_post] = 0
    p[valid & pred_water_post & pred_water_pre] = 1
    p[valid & pred_water_post & ~pred_water_pre] = 2
    return p


def _tensorboard_logdir_from_config(config: dict[str, Any]) -> Path | None:
    trainer = config.get("trainer") or {}
    logger = trainer.get("logger")
    if not logger or not isinstance(logger, dict):
        return None
    class_path = str(logger.get("class_path", ""))
    if "TensorBoard" not in class_path:
        return None
    args = logger.get("init_args") or {}
    save_dir = args.get("save_dir")
    name = args.get("name")
    version = args.get("version")
    if save_dir is None or name is None or version is None:
        return None
    return Path(str(save_dir)) / str(name) / str(version)


def _checkpoint_global_step(ckpt_path: Path) -> int | None:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        return None
    gs = ckpt.get("global_step")
    if gs is not None:
        return int(gs)
    ep = ckpt.get("epoch")
    if ep is not None:
        return int(ep)
    return None


def _finite_float(x: float) -> bool:
    return bool(torch.isfinite(torch.tensor(x, dtype=torch.float64)).item())


def _log_tensorboard_scalars(logdir: Path, global_step: int, scalars: dict[str, float]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    try:
        for tag, val in scalars.items():
            if _finite_float(val):
                writer.add_scalar(tag, float(val), global_step)
    finally:
        writer.flush()
        writer.close()
    LOG.info("TensorBoard: wrote %s under %s (step=%s)", list(scalars.keys()), logdir.resolve(), global_step)


def main(
    config_path: Path,
    ckpt_path: Path,
    tensorboard_logdir: Path,
    batch_size: int = 32,
    device: str = "cuda",
    num_workers: int = 0,
    max_batches: int | None = None,
    tensorboard_global_step: int | None = None,
) -> None:
    compute_device = device
    load_map_location: str | torch.device = device
    if (isinstance(device, str) and device.startswith("cuda")) and not torch.cuda.is_available():
        LOG.warning("CUDA requested but not available; using CPU for checkpoint load and inference.")
        compute_device = "cpu"
        load_map_location = "cpu"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    use_autocast = _use_bf16_autocast(_trainer_precision(config), compute_device)
    LOG.info("Inference autocast (bf16)=%s (trainer precision=%s)", use_autocast, _trainer_precision(config))

    data_cfg = config["data"]
    DataModuleClass = _get_class(data_cfg["class_path"])
    data_init = dict(data_cfg["init_args"])
    data_init["paired"] = True
    data_init["num_workers"] = num_workers
    data_init["batch_size"] = batch_size
    for key in ("train_transform", "val_transform", "test_transform", "predict_transform"):
        if key in data_init and data_init[key] is not None:
            data_init[key] = _resolve_transform_list(data_init[key])

    dm = DataModuleClass(**data_init)
    dm.setup("test")
    test_loader = dm.test_dataloader()
    n_batches = len(test_loader)
    LOG.info("Paired test loader: %d batches (batch_size=%s)", n_batches, batch_size)

    model_cfg = config["model"]
    TaskClass = _get_class(model_cfg["class_path"])
    task = TaskClass.load_from_checkpoint(str(ckpt_path), map_location=load_map_location, strict=True)
    task.eval()
    task.to(compute_device)
    ignore_index = _ignore_index_from_task(task)

    # Same metric classes / averaging as terratorch SemanticSegmentationTask.configure_metrics.
    iou_bin_macro = MulticlassJaccardIndex(num_classes=2, ignore_index=ignore_index, average="macro")
    f1_bin_macro = MulticlassF1Score(num_classes=2, ignore_index=ignore_index, average="macro")
    iou_mc_macro = MulticlassJaccardIndex(num_classes=3, ignore_index=ignore_index, average="macro")
    f1_mc_macro = MulticlassF1Score(num_classes=3, ignore_index=ignore_index, average="macro")
    iou_mc_none = MulticlassJaccardIndex(num_classes=3, ignore_index=ignore_index, average=None)
    for m in (iou_bin_macro, f1_bin_macro, iou_mc_macro, f1_mc_macro, iou_mc_none):
        m.to(compute_device)

    with torch.no_grad():
        it = enumerate(tqdm(test_loader, desc="Paired TB metrics"))
        for batch_idx, batch in it:
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _prepare_batch_after_transfer(batch, dm, compute_device)

            mask_pre = batch["mask_pre"].long()
            mask_post = batch["mask_post"].long()
            valid = (mask_pre != ignore_index) & (mask_post != ignore_index)

            extras_pre = _extras_for_forward_pre(batch)
            extras_post = _extras_for_forward_post(batch)
            logits_pre = _task_forward_logits(task, batch["image_pre"], extras_pre, use_autocast)
            logits_post = _task_forward_logits(task, batch["image_post"], extras_post, use_autocast)

            pred_w_pre = _pred_water_bool(logits_pre)
            pred_w_post = _pred_water_bool(logits_post)

            y_bin = _y_binary_post(mask_post, valid, ignore_index)
            p_bin = _pred_binary_post(logits_post, valid, ignore_index)
            iou_bin_macro.update(p_bin, y_bin)
            f1_bin_macro.update(p_bin, y_bin)

            y_mc = _y_three_class(mask_pre, mask_post, valid, ignore_index)
            p_mc = _pred_three_class(pred_w_pre, pred_w_post, valid, ignore_index)
            iou_mc_macro.update(p_mc, y_mc)
            f1_mc_macro.update(p_mc, y_mc)
            iou_mc_none.update(p_mc, y_mc)

    f1_bin = float(f1_bin_macro.compute().item())
    iou_bin = float(iou_bin_macro.compute().item())
    f1_mc = float(f1_mc_macro.compute().item())
    iou_mc = float(iou_mc_macro.compute().item())
    iou_per = iou_mc_none.compute()

    scalars = {
        "test/F1_Score_Binary": f1_bin,
        "test/F1_Score_MultiClass": f1_mc,
        "test/Iou_Binary": iou_bin,
        "test/IoU_MultiClass": iou_mc,
        "test/IoU_0": float(iou_per[0].item()),
        "test/IoU_1_perm": float(iou_per[1].item()),
        "test/IoU_2_flood": float(iou_per[2].item()),
    }

    LOG.info(
        "Binary  — F1: %.6f  Iou: %.6f | MultiClass — F1: %.6f  IoU: %.6f | IoU_0/perm/flood: %.6f / %.6f / %.6f",
        f1_bin,
        iou_bin,
        f1_mc,
        iou_mc,
        scalars["test/IoU_0"],
        scalars["test/IoU_1_perm"],
        scalars["test/IoU_2_flood"],
    )

    tb_step = tensorboard_global_step
    if tb_step is None:
        tb_step = _checkpoint_global_step(ckpt_path)
    if tb_step is None:
        tb_step = 0
    _log_tensorboard_scalars(tensorboard_logdir, tb_step, scalars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired EMSR binary + 3-class metrics → TensorBoard")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML (same as training)")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint .ckpt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many batches (debug / smoke test)",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="TensorBoard run directory. Default: trainer.logger init_args when TensorBoardLogger is set.",
    )
    parser.add_argument(
        "--tb-global-step",
        type=int,
        default=None,
        help="TensorBoard x-axis step. Default: checkpoint global_step, else 0.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tb_dir = args.tensorboard_logdir
    if tb_dir is None:
        tb_dir = _tensorboard_logdir_from_config(cfg)
    if tb_dir is None:
        LOG.error(
            "TensorBoard log directory is required. Pass --tensorboard-logdir or use TensorBoardLogger in config."
        )
        sys.exit(2)

    LOG.info("TensorBoard log directory: %s", tb_dir.resolve())

    main(
        config_path=args.config,
        ckpt_path=args.ckpt_path,
        tensorboard_logdir=tb_dir,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        tensorboard_global_step=args.tb_global_step,
    )

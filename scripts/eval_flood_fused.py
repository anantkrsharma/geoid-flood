#!/usr/bin/env python3
"""Fused early-fusion 3-class EMSR eval: binary (post water) and 3-class metrics → TensorBoard.

For models with a NATIVE 3-class segmentation head trained in early-fusion mode
(``fuse_paired: true``): a single forward on the fused pre+post ``image`` produces 3-class
logits directly (0 = background, 1 = permanent water, 2 = flood).

**Binary** (post water): target = ground-truth water union ``mask ∈ {1, 2}``, prediction =
``argmax(logits) ∈ {1, 2}``. This is byte-for-byte the same post-water target used by the
2-class paired pipeline (``_remap_label(..., "post")`` maps {0→0, 1→1, 2→1}), so the binary
number is directly comparable to ``scripts/eval_flood_paired.py``.

**3-class**: prediction ``argmax(logits) ∈ {0, 1, 2}`` vs the raw 3-class ``mask``. Per-class IoU
is logged as ``test/IoU_0`` (background), ``test/IoU_1_perm`` (permanent water), ``test/IoU_2_flood``.

Metrics, normalization, autocast, ignore-index resolution, and the TensorBoard tags are reused
verbatim from ``scripts/eval_flood_paired.py`` (imported, not duplicated) so results land in the
same dashboards and are computed identically to the paired comparison pipeline.

Requires ``--tensorboard-logdir`` or TensorBoardLogger paths in config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from tqdm import tqdm

# scripts/ is not a package; add it to sys.path so we can reuse the paired-eval helpers.
# Importing the module only runs its top-level defs + logging.basicConfig (its CLI is guarded by
# ``if __name__ == "__main__":``), so main() does not fire on import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_flood_paired import (  # noqa: E402
    _checkpoint_global_step,
    _get_class,
    _ignore_index_from_task,
    _log_tensorboard_scalars,
    _prepare_batch_after_transfer,
    _resolve_transform_list,
    _task_forward_logits,
    _tensorboard_logdir_from_config,
    _trainer_precision,
    _use_bf16_autocast,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


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
    # fuse_paired requires paired=True (datamodule asserts this); keep fuse_paired from the config.
    data_init["paired"] = True
    data_init["num_workers"] = num_workers
    data_init["batch_size"] = batch_size
    if not data_init.get("fuse_paired", False):
        LOG.warning(
            "Config does not set fuse_paired=True; this script expects a fused 3-class model "
            "that returns 'image'+'mask'. Proceeding, but the batch may not contain 'image'/'mask'."
        )
    for key in ("train_transform", "val_transform", "test_transform", "predict_transform"):
        if key in data_init and data_init[key] is not None:
            data_init[key] = _resolve_transform_list(data_init[key])

    dm = DataModuleClass(**data_init)
    dm.setup("test")
    test_loader = dm.test_dataloader()
    n_batches = len(test_loader)
    LOG.info("Fused test loader: %d batches (batch_size=%s)", n_batches, batch_size)

    model_cfg = config["model"]
    TaskClass = _get_class(model_cfg["class_path"])
    task = TaskClass.load_from_checkpoint(str(ckpt_path), map_location=load_map_location, strict=True)
    task.eval()
    task.to(compute_device)
    ignore_index = _ignore_index_from_task(task)

    # Same metric classes / averaging as terratorch SemanticSegmentationTask.configure_metrics
    # and scripts/eval_flood_paired.py.
    iou_bin_macro = MulticlassJaccardIndex(num_classes=2, ignore_index=ignore_index, average="macro")
    f1_bin_macro = MulticlassF1Score(num_classes=2, ignore_index=ignore_index, average="macro")
    iou_mc_macro = MulticlassJaccardIndex(num_classes=3, ignore_index=ignore_index, average="macro")
    f1_mc_macro = MulticlassF1Score(num_classes=3, ignore_index=ignore_index, average="macro")
    iou_mc_none = MulticlassJaccardIndex(num_classes=3, ignore_index=ignore_index, average=None)
    for m in (iou_bin_macro, f1_bin_macro, iou_mc_macro, f1_mc_macro, iou_mc_none):
        m.to(compute_device)

    with torch.no_grad():
        it = enumerate(tqdm(test_loader, desc="Fused TB metrics"))
        for batch_idx, batch in it:
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _prepare_batch_after_transfer(batch, dm, compute_device)

            mask = batch["mask"].long()
            valid = mask != ignore_index

            # Single forward on the fused (pre+post) image; fused samples carry no timestamps.
            logits = _task_forward_logits(task, batch["image"], {}, use_autocast)  # (B, 3, H, W)
            pred3 = logits.argmax(dim=1).long()  # (B, H, W) in {0, 1, 2}

            # 3-class metrics: pred3 vs raw mask, gated by valid.
            y_mc = torch.where(valid, mask, torch.full_like(mask, ignore_index))
            p_mc = torch.where(valid, pred3, torch.full_like(pred3, ignore_index))
            iou_mc_macro.update(p_mc, y_mc)
            f1_mc_macro.update(p_mc, y_mc)
            iou_mc_none.update(p_mc, y_mc)

            # Binary metrics: collapse water = classes {1, 2}.
            y_bin = torch.where(valid, ((mask == 1) | (mask == 2)).long(), torch.full_like(mask, ignore_index))
            p_bin = torch.where(valid, ((pred3 == 1) | (pred3 == 2)).long(), torch.full_like(pred3, ignore_index))
            iou_bin_macro.update(p_bin, y_bin)
            f1_bin_macro.update(p_bin, y_bin)

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
    parser = argparse.ArgumentParser(description="Fused early-fusion 3-class EMSR binary + 3-class metrics → TensorBoard")
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

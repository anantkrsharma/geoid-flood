"""Single-image segmentation task that forwards only real model inputs."""

from typing import Any

from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask


class FloodSegmentationTask(SemanticSegmentationTask):
    """Segmentation task that passes only model inputs to the model.

    TerraTorch's :class:`SemanticSegmentationTask` splats every batch key except ``image``,
    ``mask`` and ``filename`` into the model as keyword arguments. GEOID-Flood samples also carry
    bookkeeping fields (``chip_id``, ``split``, ``image_time``), and a backbone with a strict
    ``forward`` signature rejects them::

        TypeError: FeatureListNet.forward() got an unexpected keyword argument 'chip_id'

    That is what the timm baselines (ResNet, ConvNeXt, Swin) do. This task keeps only the keys in
    :attr:`MODEL_INPUT_KEYS` plus the three TerraTorch reads itself, so adding a metadata field to
    a dataset can never break training. If a backbone needs a new batch input, add its key to
    :attr:`MODEL_INPUT_KEYS`.

    Signature introspection would not work here: the outer model is
    ``PixelWiseModel.forward(self, x, **kwargs)``, which accepts anything and forwards it on to the
    encoder, so the strict signature is not the one that would be inspected.
    """

    #: Batch keys the model itself consumes. Everything else is bookkeeping and is dropped.
    MODEL_INPUT_KEYS = frozenset({"timestamps"})

    #: Keys TerraTorch's task reads off the batch directly (``filename`` names predictions).
    _TASK_KEYS = frozenset({"image", "mask", "filename"})

    def _model_inputs(self, batch: Any) -> dict[str, Any]:
        keep = self.MODEL_INPUT_KEYS | self._TASK_KEYS
        return {key: value for key, value in batch.items() if key in keep}

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return super().training_step(self._model_inputs(batch), batch_idx, dataloader_idx)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return super().validation_step(self._model_inputs(batch), batch_idx, dataloader_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return super().test_step(self._model_inputs(batch), batch_idx, dataloader_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return super().predict_step(self._model_inputs(batch), batch_idx, dataloader_idx)

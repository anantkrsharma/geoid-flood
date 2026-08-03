"""Early-fusion wrapper that feeds a channel-stacked fused tensor to the backbone as native modalities.

The ``fuse_paired`` data mode yields a single tensor whose channels are ordered
``[modalities_pre ... | modalities_post ...]``. Treating that whole tensor as one stacked modality
forces optical (S2) bands through the SAR (sen1grd) patch embedding. Instead, this wrapper regroups
the channels by backbone modality (concatenating the same modality across pre/post timestamps) into a
``{modality: tensor}`` dict, so the TerraMind backbone uses each modality's own *pretrained* patch
embedding. A standard encoder-decoder forward then produces the multi-class output.

The two S1 timestamps share the ``sen1grd`` key, so they must be contiguous in the fused tensor and
are concatenated into a single 4-channel ``sen1grd`` entry; S2 (pre-only) becomes a native
``sen2l2a`` entry. Order the channels via the ``combined_modalities`` spec (see below).

Heavy terratorch model symbols are imported lazily inside methods to avoid circular imports when this
module is loaded during terratorch registry init.
"""

import torch
from torch import nn


class EarlyFusionModel(nn.Module):
    """Wrap a terratorch ``PixelWiseModel`` and feed it native-modality dicts.

    Args:
        base: A terratorch model (e.g. ``EncoderDecoderFactory`` output) whose forward accepts a
            ``{modality: tensor}`` dict (the TerraMind backbone does).
        combined_modalities: Ordered ``[[backbone_key, n_channels], ...]`` describing how to slice
            the fused input tensor into modality tensors, e.g.
            ``[["sen2l2a", 10], ["sen1grd", 4]]`` for ``[S2_pre | S1_pre | S1_post]``.
    """

    def __init__(self, base: nn.Module, combined_modalities: list) -> None:
        super().__init__()
        if not combined_modalities:
            raise ValueError("EarlyFusionModel requires a non-empty combined_modalities spec.")
        self.base = base
        self.combined_modalities = combined_modalities

    def forward(self, x: torch.Tensor, **kwargs):
        image_dict: dict[str, torch.Tensor] = {}
        offset = 0
        for key, nch in self.combined_modalities:
            image_dict[str(key)] = x[:, offset : offset + int(nch)]
            offset += int(nch)
        return self.base(image_dict, **kwargs)

    def freeze_encoder(self) -> None:
        self.base.freeze_encoder()

    def freeze_decoder(self) -> None:
        self.base.freeze_decoder()

    def freeze_head(self) -> None:
        if hasattr(self.base, "freeze_head"):
            self.base.freeze_head()


class EarlyFusionEncoderDecoderFactory:
    """Build a single-encoder model that ingests a fused tensor as native TerraMind modalities.

    Accepts the same ``model_args`` as ``EncoderDecoderFactory`` plus the extra key
    ``combined_modalities`` (the slice spec used by :class:`EarlyFusionModel`). The backbone should be
    configured with the matching ``backbone_modalities`` / ``backbone_bands`` / ``backbone_in_chans``
    and ``backbone_merge_method`` (e.g. ``mean``) so the multi-modality tokens reduce to one grid.
    """

    def build_model(
        self,
        task: str,
        *,
        combined_modalities: list,
        aux_decoders=None,
        **model_args,
    ):
        from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory

        factory = EncoderDecoderFactory()
        base = factory.build_model(task, aux_decoders=aux_decoders, **model_args)
        return EarlyFusionModel(base, combined_modalities)


from terratorch.registry import MODEL_FACTORY_REGISTRY  # noqa: E402

MODEL_FACTORY_REGISTRY.register(EarlyFusionEncoderDecoderFactory)

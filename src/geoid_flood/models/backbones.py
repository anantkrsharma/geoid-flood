from typing import Any

import torch
from olmoearth_pretrain.data.constants import Modality as OlmoModality
from olmoearth_pretrain.data.constants import ModalitySpec
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from torch import Tensor, nn

# standard band orders as provided by typical data pipelines
# this is used to make olmoearth compatible with S2 L1C data.
_STANDARD_BANDS: dict[str, list[str]] = {
    "S2L2A": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    "S2L1C": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
    "S1GRD": ["vv", "vh"],
}

# maps Terramind modality names → (OlmoEarth ModalitySpec, field name in MaskedOlmoEarthSample)
_MODALITY_CONFIG: dict[str, tuple[ModalitySpec, str]] = {
    "S2L2A": (OlmoModality.SENTINEL2_L2A, "sentinel2_l2a"),
    "S2L1C": (OlmoModality.SENTINEL2_L2A, "sentinel2_l2a"),
    "S1GRD": (OlmoModality.SENTINEL1, "sentinel1"),
}


class ModalityHandler:
    """Precomputed band reordering and normalization config for a single modality."""

    def __init__(self, modality: str, bands: list[str] | None = None) -> None:
        spec, self.field = _MODALITY_CONFIG[modality]
        self.num_band_sets = len(spec.band_sets)

        source_bands = bands or _STANDARD_BANDS[modality]
        if modality == "S2L1C":
            self.drop_b10 = [i for i, b in enumerate(source_bands) if b != "B10"]
            source_bands = [b for b in source_bands if b != "B10"]
        else:
            self.drop_b10 = None
        self.reorder = [source_bands.index(b) for b in spec.band_order]

        # normalization: computed mean/std (std_multiplier=2) → [0, 1]
        normalizer = Normalizer(strategy=Strategy.COMPUTED)
        norm_cfg = normalizer.norm_config[spec.name]
        target_bands = spec.band_order
        means = torch.tensor([norm_cfg[b]["mean"] for b in target_bands], dtype=torch.float32)
        stds = torch.tensor([norm_cfg[b]["std"] for b in target_bands], dtype=torch.float32)
        self.norm_min = means - 2 * stds
        self.norm_range = 4 * stds

    def prepare(self, x: Tensor) -> Tensor:
        """Band selection, reorder and normalize. [B, C, H, W] → [B, C', H, W]."""
        if self.drop_b10 is not None:
            x = x[:, self.drop_b10]
        x = x[:, self.reorder]
        return (x - self.norm_min.to(x)[None, :, None, None]) / self.norm_range.to(x)[None, :, None, None]


class OlmoEarthBackbone(nn.Module):
    """Wraps OlmoEarth's encoder into a terratorch-compatible backbone interface.

    Handles band reordering, normalization, and the conversion from standard
    [B, C, H, W] tensors to OlmoEarth's MaskedOlmoEarthSample format.
    Supports single or multiple modalities (e.g. S2 only, S1 only, or S2+S1).
    """

    def __init__(
        self,
        name: ModelID,
        modalities: tuple[str, ...],
        pretrained: bool = True,
        patch_size: int = 4,
        bands: dict[str, list[str]] | None = None,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        bands = bands or {}
        self.modalities = modalities
        self.patch_size = patch_size
        self._handlers = {mod: ModalityHandler(mod, bands.get(mod)) for mod in modalities}

        model = load_model_from_id(model_id=name, load_weights=pretrained)
        self.encoder: nn.Module = model.encoder  # type: ignore[assignment]

        blocks: nn.ModuleList = self.encoder.blocks  # type: ignore[assignment]
        depth = len(blocks)
        resolved = out_indices or list(range(depth))
        self._out_indices = sorted(i % depth for i in resolved)
        self._intermediate: dict[int, Tensor] = {}
        for idx in self._out_indices:
            blocks[idx].register_forward_hook(self._capture_hook(idx))

        # infer embedding dim from the first block output (all blocks same dim)
        self._embed_dim = self.encoder.blocks[0].mlp.fc1.in_features  # type: ignore[union-attr]
        # the total out_channels size is given from the actual out (e.g. 128) * num_band_sets (e.g. 3 for S2)
        # example: 4-skips olmoearth, out_channels = [384, 384, 384, 384] (128x3)
        self.total_band_sets = sum(handler.num_band_sets for handler in self._handlers.values())
        self.out_channels = [self._embed_dim * self.total_band_sets] * len(self._out_indices)

    def _capture_hook(self, idx: int):
        def hook(module: nn.Module, input: Any, output: Tensor) -> None:
            self._intermediate[idx] = output

        return hook

    def _build_sample(self, inputs: dict[str, Tensor]) -> MaskedOlmoEarthSample:
        """Convert input tensors to MaskedOlmoEarthSample."""
        kwargs: dict[str, Tensor] = {}
        ref_tensor = next(iter(inputs.values()))

        for mod in self.modalities:
            handler = self._handlers[mod]
            x = handler.prepare(inputs[mod])  # [B, C, H, W]
            x = x.permute(0, 2, 3, 1).unsqueeze(3)  # [B, H, W, T=1, C]
            b, h, w, t, _ = x.shape

            mask = torch.full(
                (b, h, w, t, handler.num_band_sets),
                MaskValue.ONLINE_ENCODER.value,
                device=x.device,
                dtype=x.dtype,
            )
            kwargs[handler.field] = x
            kwargs[f"{handler.field}_mask"] = mask

        b = ref_tensor.shape[0]
        # timestamps: [B, T=1, 3] with [day (1-31), month (0-11), year]
        if "timestamps" in inputs:
            ts = inputs["timestamps"].float()  # [B, 3]
            timestamps = ts.unsqueeze(1) if ts.ndim == 2 else ts  # ensure [B, T, 3]
        else:
            timestamps = torch.tensor([[1, 6, 2024]], device=ref_tensor.device, dtype=torch.float32).expand(b, 1, 3)
        return MaskedOlmoEarthSample(timestamps=timestamps.long(), **kwargs)

    def forward(self, d: dict[str, Tensor], **kwargs) -> list[Tensor]:
        """Run encoder and return intermediate layer features.

        Args:
            inputs: mapping of modality name → [B, C, H, W] tensor.

        Returns:
            list of feature tensors [B, D, H_p, W_p], one per requested ``out_indices`` layer.
        """
        if isinstance(d, Tensor):
            d = {"S1GRD": d}
        self._intermediate.clear()
        ref = next(iter(d.values()))
        b, _, h, w = ref.shape
        h_p, w_p = h // self.patch_size, w // self.patch_size

        sample = self._build_sample(d)
        self.encoder(sample, patch_size=self.patch_size, fast_pass=True)
        return [self._intermediate[idx].reshape(b, h_p, w_p, -1).permute(0, 3, 1, 2) for idx in self._out_indices]


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_base(pretrained: bool = True, **kwargs):
    return OlmoEarthBackbone(name=ModelID.OLMOEARTH_V1_BASE, pretrained=pretrained, **kwargs)

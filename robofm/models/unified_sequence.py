"""Task-agnostic language/image causal sequence model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from robofm.backbones import BackboneConfig, build_backbone
from robofm.dataio.schema import AtomType
from robofm.dataio.torch_batch import TorchTargetBatch, TorchUnifiedBatch


@dataclass(frozen=True)
class UnifiedModelConfig:
    vocab_size: int
    hidden_size: int
    backbone: BackboneConfig | Mapping[str, Any]
    image_channels: int = 3
    image_width: int = 64
    image_target_weight: float = 1.0
    target_mode: str = "causal"
    detach_image_targets: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 1 or self.hidden_size < 1:
            raise ValueError("vocab_size and hidden_size must be positive")
        if self.target_mode not in {"causal", "explicit"}:
            raise ValueError("target_mode must be 'causal' or 'explicit'")


@dataclass
class UnifiedModelOutput:
    hidden_states: torch.Tensor
    state: Any
    language_loss_sum: torch.Tensor
    image_loss_sum: torch.Tensor
    language_weight: torch.Tensor
    image_weight: torch.Tensor
    metrics: Mapping[str, Any]

    @property
    def loss_sum(self) -> torch.Tensor:
        return self.language_loss_sum + self.image_loss_sum

    @property
    def loss_weight(self) -> torch.Tensor:
        return self.language_weight + self.image_weight

    @property
    def loss(self) -> torch.Tensor:
        return self.loss_sum / self.loss_weight.clamp_min(1.0)


class RawImageEncoder(nn.Module):
    """Small fixed-channel encoder mapping one raw image to one sequence atom."""

    def __init__(self, channels: int, hidden_size: int, width: int = 64):
        super().__init__()
        if channels < 1 or width < 1:
            raise ValueError("image channels and width must be positive")
        self.channels = channels
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, hidden_size),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            if image.shape[0] == self.channels:
                image = image.unsqueeze(0)
            elif image.shape[-1] == self.channels:
                image = image.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"image has no channel dimension of size {self.channels}")
        elif image.ndim != 4:
            raise ValueError("raw image must be CHW, HWC, BCHW, or BHWC")
        elif image.shape[1] != self.channels and image.shape[-1] == self.channels:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} image channels, got {image.shape[1]}")
        if image.dtype == torch.uint8:
            image = image.to(torch.float32).div_(255.0)
        else:
            image = image.to(torch.float32)
        parameter = next(self.parameters())
        return self.network(image.to(device=parameter.device, dtype=parameter.dtype))


class UnifiedSequenceModel(nn.Module):
    """A single causal model whose persistent atoms are language or image only."""

    def __init__(self, config: UnifiedModelConfig,
                 image_encoder: nn.Module | None = None):
        super().__init__()
        self.config = config
        backbone_config = (
            config.backbone if isinstance(config.backbone, BackboneConfig)
            else BackboneConfig.from_mapping(config.backbone)
        )
        if backbone_config.hidden_size != config.hidden_size:
            raise ValueError("model and backbone hidden_size must match")
        self.language_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.type_embedding = nn.Embedding(3, config.hidden_size, padding_idx=0)
        self.image_encoder = image_encoder or RawImageEncoder(
            config.image_channels, config.hidden_size, config.image_width
        )
        self.backbone = build_backbone(backbone_config)
        self.language_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_head = nn.Linear(config.hidden_size, config.hidden_size)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(image)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        if encoded.shape != (1, self.config.hidden_size):
            raise ValueError(
                "image encoder must return [1, hidden_size] for each image atom; "
                f"got {tuple(encoded.shape)}"
            )
        return encoded[0]

    def _embed_atoms(self, batch: TorchUnifiedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch.atom_values
        types = batch.atom_types
        language_mask = (types == int(AtomType.LANGUAGE_TOKEN)) & batch.valid_atom_mask
        image_mask = (types == int(AtomType.IMAGE)) & batch.valid_atom_mask
        known = language_mask | image_mask | ~batch.valid_atom_mask
        if not bool(torch.all(known)):
            unknown = torch.unique(types[~known]).tolist()
            raise ValueError(f"unsupported atom types in batch: {unknown}")
        if bool(torch.any(language_mask & (values >= self.config.vocab_size))):
            invalid = int(values[language_mask].max())
            raise ValueError(f"language token ID {invalid} exceeds vocab_size={self.config.vocab_size}")
        embeddings = torch.zeros(
            (*values.shape, self.config.hidden_size),
            device=values.device,
            dtype=self.language_embedding.weight.dtype,
        )
        if bool(torch.any(language_mask)):
            embeddings[language_mask] = self.language_embedding(values[language_mask])
        image_latents = torch.zeros_like(embeddings)
        for lane, positions in enumerate(batch.image_payloads):
            for position, image in positions.items():
                if not bool(image_mask[lane, position]):
                    raise ValueError(f"image payload at non-image atom lane={lane}, position={position}")
                latent = self._encode_image(image)
                embeddings[lane, position] = latent
                image_latents[lane, position] = latent
        expected_images = int(image_mask.sum().item())
        supplied_images = sum(len(images) for images in batch.image_payloads)
        if supplied_images != expected_images:
            raise ValueError(
                f"missing image payloads: expected {expected_images}, received {supplied_images}"
            )
        embeddings = embeddings + self.type_embedding(types.clamp(min=0, max=2))
        embeddings = embeddings * batch.valid_atom_mask.unsqueeze(-1)
        return embeddings, image_latents

    def _language_loss(self, hidden: torch.Tensor, target: torch.Tensor,
                       weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.numel() == 0:
            zero = hidden.sum()
            return zero, weights.sum()
        if bool(torch.any((target < 0) | (target >= self.config.vocab_size))):
            raise ValueError("language target is outside the configured vocabulary")
        loss = F.cross_entropy(self.language_head(hidden), target, reduction="none")
        return (loss * weights).sum(), weights.sum()

    def _image_loss(self, hidden: torch.Tensor, targets: torch.Tensor,
                    weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.numel() == 0:
            zero = hidden.sum()
            return zero, weights.sum()
        if self.config.detach_image_targets:
            targets = targets.detach()
        per_target = F.mse_loss(self.image_head(hidden), targets, reduction="none").mean(dim=-1)
        weighted = weights * self.config.image_target_weight
        return (per_target * weighted).sum(), weighted.sum()

    def _causal_losses(self, hidden: torch.Tensor, image_latents: torch.Tensor,
                       batch: TorchUnifiedBatch, pending: Any) -> tuple[torch.Tensor, ...]:
        batch_size, length, hidden_size = hidden.shape
        source_parts = []
        type_parts = []
        value_parts = []
        image_parts = []
        weight_parts = []
        if pending is not None and length:
            bridge_valid = (
                pending["valid"] & batch.valid_atom_mask[:, 0] &
                batch.loss_mask[:, 0] & ~batch.reset_mask[:, 0]
            )
            source_parts.append(pending["hidden"])
            type_parts.append(batch.atom_types[:, 0])
            value_parts.append(batch.atom_values[:, 0])
            image_parts.append(image_latents[:, 0])
            weight_parts.append(bridge_valid.to(hidden.dtype))
        if length > 1:
            pair_valid = (
                batch.valid_atom_mask[:, :-1] & batch.valid_atom_mask[:, 1:] &
                batch.loss_mask[:, 1:] & ~batch.reset_mask[:, 1:]
            )
            source_parts.append(hidden[:, :-1].reshape(-1, hidden_size))
            type_parts.append(batch.atom_types[:, 1:].reshape(-1))
            value_parts.append(batch.atom_values[:, 1:].reshape(-1))
            image_parts.append(image_latents[:, 1:].reshape(-1, hidden_size))
            weight_parts.append(pair_valid.reshape(-1).to(hidden.dtype))
        if not source_parts:
            zero = hidden.sum()
            return zero, zero, zero.detach(), zero.detach()
        sources = torch.cat([part.reshape(-1, hidden_size) for part in source_parts])
        target_types = torch.cat([part.reshape(-1) for part in type_parts])
        target_values = torch.cat([part.reshape(-1) for part in value_parts])
        target_images = torch.cat([part.reshape(-1, hidden_size) for part in image_parts])
        weights = torch.cat([part.reshape(-1) for part in weight_parts])
        language = (target_types == int(AtomType.LANGUAGE_TOKEN)) & (weights > 0)
        image = (target_types == int(AtomType.IMAGE)) & (weights > 0)
        language_loss, language_weight = self._language_loss(
            sources[language], target_values[language], weights[language]
        )
        image_loss, image_weight = self._image_loss(
            sources[image], target_images[image], weights[image]
        )
        return language_loss, image_loss, language_weight, image_weight

    def _explicit_losses(self, hidden: torch.Tensor, targets: TorchTargetBatch,
                         loss_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if targets.source_positions.numel() == 0:
            zero = hidden.sum()
            return zero, zero, zero.detach(), zero.detach()
        lanes = targets.lane_indices
        positions = targets.source_positions
        in_bounds = positions < hidden.shape[1]
        safe_positions = positions.clamp(max=max(hidden.shape[1] - 1, 0))
        valid = targets.valid & in_bounds & loss_mask[lanes, safe_positions]
        weights = targets.weights.to(hidden.dtype) * valid.to(hidden.dtype)
        sources = hidden[lanes, safe_positions]
        language = (targets.atom_types == int(AtomType.LANGUAGE_TOKEN)) & valid
        language_loss, language_weight = self._language_loss(
            sources[language], targets.target_values[language], weights[language]
        )
        image_indices = torch.nonzero(
            (targets.atom_types == int(AtomType.IMAGE)) & valid, as_tuple=False
        ).flatten()
        image_sources = sources.index_select(0, image_indices)
        image_latents = []
        image_weights = []
        for target_index in image_indices.tolist():
            try:
                payload = targets.image_payloads[target_index]
            except KeyError as error:
                raise ValueError(f"missing explicit image target payload {target_index}") from error
            image_latents.append(self._encode_image(payload))
            image_weights.append(weights[target_index])
        if image_latents:
            image_loss, image_weight = self._image_loss(
                image_sources, torch.stack(image_latents), torch.stack(image_weights)
            )
        else:
            image_loss = hidden.sum() * 0.0
            image_weight = weights.new_zeros(())
        return language_loss, image_loss, language_weight, image_weight

    def forward_chunk(self, batch: TorchUnifiedBatch, *, state: Any = None,
                      return_state: bool = True) -> UnifiedModelOutput:
        embeddings, image_latents = self._embed_atoms(batch)
        backbone_state = None if state is None else state.get("backbone")
        pending = None if state is None else state.get("pending")
        backbone_output = self.backbone.forward_chunk(
            embeddings,
            state=backbone_state,
            attention_mask=batch.attention_mask,
            memory_update_mask=batch.memory_update_mask,
            reset_mask=batch.reset_mask,
            position_ids=batch.positions,
            return_state=return_state,
        )
        hidden = backbone_output.hidden_states
        if self.config.target_mode == "causal":
            losses = self._causal_losses(hidden, image_latents, batch, pending)
        else:
            losses = self._explicit_losses(hidden, batch.targets, batch.loss_mask)

        if hidden.shape[1]:
            lane_indices = torch.arange(hidden.shape[0], device=hidden.device)
            last_indices = (batch.lengths - 1).clamp_min(0)
            pending_hidden = hidden[lane_indices, last_indices]
        else:
            pending_hidden = hidden.new_zeros((hidden.shape[0], hidden.shape[2]))
        new_pending = {"hidden": pending_hidden, "valid": batch.lengths > 0}
        if pending is not None:
            inactive = batch.lengths == 0
            new_pending["hidden"] = torch.where(
                inactive.unsqueeze(-1), pending["hidden"], new_pending["hidden"]
            )
            new_pending["valid"] = torch.where(inactive, pending["valid"], new_pending["valid"])
        next_state = {
            "backbone": backbone_output.state,
            "pending": new_pending,
        } if return_state else None
        return UnifiedModelOutput(
            hidden_states=hidden,
            state=next_state,
            language_loss_sum=losses[0],
            image_loss_sum=losses[1],
            language_weight=losses[2],
            image_weight=losses[3],
            metrics=backbone_output.metrics,
        )

    def forward(self, batch: TorchUnifiedBatch, **kwargs: Any) -> UnifiedModelOutput:
        return self.forward_chunk(batch, **kwargs)

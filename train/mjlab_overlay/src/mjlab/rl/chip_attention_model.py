from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


def _activation(name: str) -> nn.Module:
  return resolve_nn_activation(name)


def _attention_defaults() -> dict[str, Any]:
  return {
    "history_length": 10,
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 512,
    "dropout": 0.0,
    "store_attention_weights": False,
    "layout": {
      "command_dim": 30,
      "motion_anchor_pos_dim": 3,
      "motion_anchor_ori_dim": 6,
      "base_lin_vel_dim": 3,
      "base_ang_vel_dim": 3,
      "joint_pos_dim": 29,
      "joint_vel_dim": 29,
      "actions_dim": 29,
      "compliance_dim": 3,
      "reference_keypoint_pos_dim": 9,
      "reference_keypoint_ori_dim": 18,
    },
  }


def _merge_attention_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
  merged = _attention_defaults()
  if cfg is None:
    return merged
  merged.update({k: v for k, v in cfg.items() if k != "layout"})
  layout = dict(merged["layout"])
  layout.update(cfg.get("layout", {}))
  merged["layout"] = layout
  return merged


class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int) -> None:
    super().__init__()
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2, dtype=torch.float32)
      * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class ChipTemporalCrossAttentionCore(nn.Module):
  """Encode CHIP proprioceptive/action history with cross-attention.

  The environment still exposes the original flat CHIP observation layout. This
  module reconstructs the 10-step history sequence internally, preserving
  deployment input compatibility while avoiding an all-flat MLP encoder.
  """

  def __init__(
    self,
    obs_dim: int,
    attention_cfg: dict[str, Any] | None,
    activation: str,
  ) -> None:
    super().__init__()
    cfg = _merge_attention_cfg(attention_cfg)
    layout = cfg["layout"]

    self.history_length = int(cfg["history_length"])
    self.embed_dim = int(cfg["embed_dim"])
    self.store_attention_weights = bool(cfg["store_attention_weights"])

    self.prefix_dim = (
      int(layout["command_dim"])
      + int(layout["motion_anchor_pos_dim"])
      + int(layout["motion_anchor_ori_dim"])
    )
    self.history_step_dims = (
      int(layout["base_lin_vel_dim"]),
      int(layout["base_ang_vel_dim"]),
      int(layout["joint_pos_dim"]),
      int(layout["joint_vel_dim"]),
      int(layout["actions_dim"]),
    )
    self.history_step_dim = sum(self.history_step_dims)
    self.history_flat_dims = tuple(d * self.history_length for d in self.history_step_dims)
    self.history_flat_dim = sum(self.history_flat_dims)
    self.tail_dim = (
      int(layout["compliance_dim"])
      + int(layout["reference_keypoint_pos_dim"])
      + int(layout["reference_keypoint_ori_dim"])
    )
    self.base_obs_dim = self.prefix_dim + self.history_flat_dim + self.tail_dim
    if obs_dim < self.base_obs_dim:
      raise ValueError(
        f"CHIP attention expected at least {self.base_obs_dim} dims, got {obs_dim}."
      )

    self.extra_context_dim = obs_dim - self.base_obs_dim
    context_dim = self.prefix_dim + self.tail_dim + self.extra_context_dim

    self.history_proj = nn.Linear(self.history_step_dim, self.embed_dim)
    self.pos_encoding = SinusoidalPositionalEncoding(self.embed_dim, self.history_length)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=self.embed_dim,
      nhead=int(cfg["num_heads"]),
      dim_feedforward=int(cfg["ff_dim"]),
      dropout=float(cfg["dropout"]),
      activation="gelu",
      batch_first=True,
    )
    self.history_encoder = nn.TransformerEncoder(
      encoder_layer,
      num_layers=int(cfg["num_layers"]),
    )

    self.context_proj = nn.Sequential(
      nn.Linear(context_dim, self.embed_dim),
      _activation(activation),
      nn.Linear(self.embed_dim, self.embed_dim),
    )
    self.cross_attention = nn.MultiheadAttention(
      embed_dim=self.embed_dim,
      num_heads=int(cfg["num_heads"]),
      dropout=float(cfg["dropout"]),
      batch_first=True,
    )
    self.fusion_norm = nn.LayerNorm(self.embed_dim)
    self.fusion = nn.Sequential(
      nn.Linear(self.embed_dim * 3, self.embed_dim),
      _activation(activation),
      nn.Linear(self.embed_dim, self.embed_dim),
      _activation(activation),
    )
    self.last_attention_weights: torch.Tensor | None = None

  def _split_flat(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = flat_obs[:, : self.prefix_dim]
    offset = self.prefix_dim

    histories = []
    for dim, flat_dim in zip(self.history_step_dims, self.history_flat_dims, strict=True):
      values = flat_obs[:, offset : offset + flat_dim]
      histories.append(values.reshape(flat_obs.shape[0], self.history_length, dim))
      offset += flat_dim
    history = torch.cat(histories, dim=-1)

    tail = flat_obs[:, offset : offset + self.tail_dim]
    offset += self.tail_dim
    extra = flat_obs[:, offset:]
    context = torch.cat((prefix, tail, extra), dim=-1)
    return history, context

  def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
    history, context = self._split_flat(flat_obs)
    history_tokens = self.history_proj(history)
    history_tokens = self.pos_encoding(history_tokens)
    history_tokens = self.history_encoder(history_tokens)

    context_token = self.context_proj(context)
    attended, weights = self.cross_attention(
      context_token.unsqueeze(1),
      history_tokens,
      history_tokens,
      need_weights=self.store_attention_weights,
      average_attn_weights=False,
    )
    if self.store_attention_weights:
      self.last_attention_weights = weights.detach()
    else:
      self.last_attention_weights = None

    attended = attended.squeeze(1)
    fused = self.fusion_norm(context_token + attended)
    pooled = history_tokens.mean(dim=1)
    recent = history_tokens[:, -1]
    return self.fusion(torch.cat((fused, recent, pooled), dim=-1))


class ChipTemporalCrossAttentionModel(MLPModel):
  """RSL-RL model for CHIP tracking with temporal cross-attention."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    attention_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._attention_cfg = _merge_attention_cfg(attention_cfg)
    self._attention_latent_dim = int(self._attention_cfg["embed_dim"])
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.attention_core = ChipTemporalCrossAttentionCore(
      obs_dim=self.obs_dim,
      attention_cfg=self._attention_cfg,
      activation=activation,
    )

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    obs_list = [obs[obs_group] for obs_group in self.obs_groups]
    flat_obs = torch.cat(obs_list, dim=-1)
    flat_obs = self.obs_normalizer(flat_obs)
    return self.attention_core(flat_obs)

  def _get_latent_dim(self) -> int:
    return self._attention_latent_dim

  def as_jit(self) -> nn.Module:
    return _TorchChipTemporalCrossAttentionModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxChipTemporalCrossAttentionModel(self, verbose)


class _ExportChipTemporalCrossAttentionModel(nn.Module):
  is_recurrent: bool = False

  def __init__(self, model: ChipTemporalCrossAttentionModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.attention_core = copy.deepcopy(model.attention_core)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    latent = self.attention_core(x)
    out = self.mlp(latent)
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class _TorchChipTemporalCrossAttentionModel(_ExportChipTemporalCrossAttentionModel):
  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxChipTemporalCrossAttentionModel(_ExportChipTemporalCrossAttentionModel):
  pass

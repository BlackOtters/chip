from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  sample_uniform,
  subtract_frame_transforms,
)

from .commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
  from typing import Any

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class ChipMotionCommand(MotionCommand):
  """Motion command that exposes CHIP hindsight targets to the policy."""

  cfg: ChipMotionCommandCfg

  def __init__(self, cfg: ChipMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if len(cfg.command_joint_names) == 0:
      raise ValueError(
        "ChipMotionCommandCfg.command_joint_names must not be empty."
      )
    self.command_joint_indexes = torch.tensor(
      self.robot.find_joints(cfg.command_joint_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.eef_body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.eef_body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.eef_motion_indexes = torch.tensor(
      [cfg.body_names.index(name) for name in cfg.eef_body_names],
      dtype=torch.long,
      device=self.device,
    )
    self.compliance = torch.zeros(
      self.num_envs, len(cfg.eef_body_names), 1, device=self.device
    )
    self.eef_force_w = torch.zeros(
      self.num_envs, len(cfg.eef_body_names), 3, device=self.device
    )
    self.eef_goal_pos_w = torch.zeros_like(self.eef_force_w)
    self.eef_hindsight_goal_pos_w = torch.zeros_like(self.eef_force_w)
    self.metrics["chip_compliance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["chip_force"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["chip_hindsight_offset"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self.joint_pos[:, self.command_joint_indexes],
        self.joint_vel[:, self.command_joint_indexes],
      ],
      dim=1,
    )

  @property
  def eef_reference_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.eef_motion_indexes]

  @property
  def eef_reference_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.eef_motion_indexes]

  @property
  def robot_eef_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.eef_body_indexes]

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self._sample_compliance(env_ids)
    self._update_chip_goals(env_ids=env_ids, initialize=True)
    self._apply_hindsight_body_targets(env_ids)

  def _update_command(self):
    super()._update_command()
    self._update_chip_goals()
    self._apply_hindsight_body_targets()

  def _sample_compliance(self, env_ids: torch.Tensor) -> None:
    low, high = self.cfg.compliance_range
    self.compliance[env_ids] = sample_uniform(
      low, high, (len(env_ids), len(self.cfg.eef_body_names), 1), self.device
    )

  def _update_chip_goals(
    self, env_ids: torch.Tensor | None = None, initialize: bool = False
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return

    force = self.robot.data.body_external_force[env_ids][:, self.eef_body_indexes]
    force_norm = torch.norm(force, dim=-1, keepdim=True)
    force_dir = force / force_norm.clamp_min(1.0e-6)
    capped_norm = force_norm.clamp(max=self.cfg.force_clip)
    force = force_dir * capped_norm
    quiet = force_norm < self.cfg.min_force
    force = torch.where(quiet, torch.zeros_like(force), force)

    reference = self.eef_reference_pos_w[env_ids]
    current = self.robot_eef_pos_w[env_ids]
    alpha = self.cfg.goal_smoothing_alpha
    comply = alpha * current + (1.0 - alpha) * self.eef_goal_pos_w[env_ids]
    recover = alpha * reference + (1.0 - alpha) * self.eef_goal_pos_w[env_ids]
    target = torch.where(quiet, recover, comply)
    if initialize:
      target = reference

    displacement = force * self.compliance[env_ids]
    displacement_norm = torch.norm(displacement, dim=-1, keepdim=True)
    displacement_scale = self.cfg.max_goal_displacement / displacement_norm.clamp_min(
      self.cfg.max_goal_displacement
    )
    displacement = displacement * displacement_scale.clamp(max=1.0)
    hindsight = target - displacement

    self.eef_force_w[env_ids] = force
    self.eef_hindsight_goal_pos_w[env_ids] = hindsight
    self.eef_goal_pos_w[env_ids] = target

  def _apply_hindsight_body_targets(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    target_ids: torch.Tensor | slice = slice(None) if env_ids is None else env_ids
    target_pos_w = self.body_pos_relative_w[target_ids].clone()
    target_pos_w[:, self.eef_motion_indexes] = self.eef_hindsight_goal_pos_w[target_ids]
    self.body_pos_relative_w[target_ids] = target_pos_w

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["chip_compliance"] = self.compliance.squeeze(-1).mean(dim=-1)
    self.metrics["chip_force"] = torch.norm(self.eef_force_w, dim=-1).mean(dim=-1)
    self.metrics["chip_hindsight_offset"] = torch.norm(
      self.eef_hindsight_goal_pos_w - self.eef_goal_pos_w, dim=-1
    ).mean(dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    reset_ids: torch.Tensor | slice = slice(None) if env_ids is None else env_ids
    self.eef_force_w[reset_ids] = 0.0
    self.eef_goal_pos_w[reset_ids] = self.eef_reference_pos_w[reset_ids]
    self.eef_hindsight_goal_pos_w[reset_ids] = self.eef_goal_pos_w[reset_ids]
    self._apply_hindsight_body_targets(reset_ids)
    return extras

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)
    if not self.cfg.viz.show_chip_targets:
      return

    for env_idx in visualizer.get_env_indices(self.num_envs):
      for i, body_name in enumerate(self.cfg.eef_body_names):
        visualizer.add_sphere(
          center=self.eef_hindsight_goal_pos_w[env_idx, i].cpu().numpy(),
          radius=0.035,
          color=self.cfg.viz.hindsight_goal_color,
          label=f"chip_hindsight_{body_name}_{env_idx}",
        )
        visualizer.add_sphere(
          center=self.eef_goal_pos_w[env_idx, i].cpu().numpy(),
          radius=0.028,
          color=self.cfg.viz.smoothed_goal_color,
          label=f"chip_goal_{body_name}_{env_idx}",
        )


@dataclass(kw_only=True)
class ChipMotionCommandCfg(MotionCommandCfg):
  eef_body_names: tuple[str, ...]
  command_joint_names: tuple[str, ...] = ()
  compliance_range: tuple[float, float] = (0.0, 0.15)
  force_clip: float = 40.0
  min_force: float = 1.0
  max_goal_displacement: float = 0.5
  goal_smoothing_alpha: float = 0.2

  @dataclass
  class VizCfg(MotionCommandCfg.VizCfg):
    show_chip_targets: bool = True
    hindsight_goal_color: tuple[float, float, float, float] = (0.95, 0.35, 0.1, 0.9)
    smoothed_goal_color: tuple[float, float, float, float] = (0.1, 0.45, 1.0, 0.9)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ChipMotionCommand:
    return ChipMotionCommand(self, env)


def _get_chip_command(env: ManagerBasedRlEnv, command_name: str) -> ChipMotionCommand:
  return cast(ChipMotionCommand, env.command_manager.get_term(command_name))


def chip_compliance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_chip_command(env, command_name)
  return command.compliance.squeeze(-1)


def chip_eef_force_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_chip_command(env, command_name)
  anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
    -1, len(command.cfg.eef_body_names), -1
  )
  force_b = quat_apply_inverse(
    anchor_quat.reshape(-1, 4),
    command.eef_force_w.reshape(-1, 3),
  ).reshape(env.num_envs, len(command.cfg.eef_body_names), 3)
  return force_b.reshape(env.num_envs, -1)



def chip_reference_keypoint_pos_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _get_chip_command(env, command_name)
  num_keypoints = len(command.cfg.eef_body_names)
  root_pos_w = command.robot.data.root_link_pos_w[:, None, :].repeat(1, num_keypoints, 1)
  root_quat_w = command.robot.data.root_link_quat_w[:, None, :].repeat(
    1, num_keypoints, 1
  )
  pos_b, _ = subtract_frame_transforms(
    root_pos_w,
    root_quat_w,
    command.eef_hindsight_goal_pos_w,
    command.eef_reference_quat_w,
  )
  return pos_b.reshape(env.num_envs, -1)


def chip_reference_keypoint_ori_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _get_chip_command(env, command_name)
  num_keypoints = len(command.cfg.eef_body_names)
  root_pos_w = command.robot.data.root_link_pos_w[:, None, :].repeat(1, num_keypoints, 1)
  root_quat_w = command.robot.data.root_link_quat_w[:, None, :].repeat(
    1, num_keypoints, 1
  )
  _, ori_b = subtract_frame_transforms(
    root_pos_w,
    root_quat_w,
    command.eef_reference_pos_w,
    command.eef_reference_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(env.num_envs, -1)


class apply_chip_eef_impulse:
  """Apply CHIP-style trapezoid random forces to selected end-effectors."""

  @dataclass
  class VizCfg:
    rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.8, 0.9)
    scale: float = 0.005
    width: float = 0.015
    min_force: float = 1.0

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._cooldown_s = cfg.params["cooldown_s"]
    self._selected_body_ids = self._resolve_body_ids(self._body_ids)
    self._num_bodies = int(self._selected_body_ids.numel())
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._duration = torch.ones(self._num_envs, device=self._device)
    self._peak_forces = torch.zeros(
      self._num_envs, self._num_bodies, 3, device=self._device
    )
    self._cooldown_remaining = self._sample_range(
      self._cooldown_s, (self._num_envs,)
    )
    self._viz_cfg = cfg.params.get("viz_cfg", apply_chip_eef_impulse.VizCfg())

  def _sample_range(self, value_range: tuple[float, float], shape) -> torch.Tensor:
    low, high = value_range
    return torch.rand(shape, device=self._device) * (high - low) + low

  def _resolve_body_ids(
    self, body_ids: torch.Tensor | slice | list[int] | tuple[int, ...] | None
  ) -> torch.Tensor:
    if body_ids is None:
      return torch.arange(self._asset.num_bodies, device=self._device, dtype=torch.long)
    if isinstance(body_ids, slice):
      return torch.arange(self._asset.num_bodies, device=self._device, dtype=torch.long)[
        body_ids
      ]
    if isinstance(body_ids, torch.Tensor):
      ids = body_ids.to(device=self._device)
      if ids.dtype == torch.bool:
        return ids.flatten().nonzero(as_tuple=False).squeeze(-1).to(torch.long)
      return ids.flatten().to(torch.long)
    return torch.tensor(body_ids, device=self._device, dtype=torch.long)

  def _resolve_env_ids(
    self, env_ids: torch.Tensor | slice | None
  ) -> torch.Tensor:
    if env_ids is None:
      return torch.arange(self._num_envs, device=self._device, dtype=torch.long)
    if isinstance(env_ids, slice):
      return torch.arange(self._num_envs, device=self._device, dtype=torch.long)[env_ids]
    ids = env_ids.to(device=self._device)
    if ids.dtype == torch.bool:
      return ids.flatten().nonzero(as_tuple=False).squeeze(-1).to(torch.long)
    return ids.flatten().to(torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    horizontal_only: bool = False,
  ) -> None:
    del env, asset_cfg
    dt = self._step_dt
    step_ids = self._resolve_env_ids(env_ids)
    if step_ids.numel() == 0:
      return
    step_mask = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    step_mask[step_ids] = True

    active_mask = self._active & step_mask
    self._time_remaining[active_mask] -= dt
    expired = active_mask & (self._time_remaining <= 0.0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      zeros = torch.zeros(
        (int(expired_ids.numel()), self._num_bodies, 3), device=self._device
      )
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
      )
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      self._peak_forces[expired_ids] = 0.0
      self._cooldown_remaining[expired_ids] = self._sample_range(
        cooldown_s, (int(expired_ids.numel()),)
      )

    active_ids = (self._active & step_mask).nonzero(as_tuple=False).squeeze(-1)
    if active_ids.numel() > 0:
      progress = 1.0 - self._time_remaining[active_ids] / self._duration[
        active_ids
      ].clamp_min(1.0e-6)
      envelope = torch.ones_like(progress)
      ramp_up = progress < 0.25
      ramp_down = progress > 0.75
      envelope[ramp_up] = progress[ramp_up] / 0.25
      envelope[ramp_down] = (1.0 - progress[ramp_down]) / 0.25
      envelope = envelope.clamp_(min=0.0, max=1.0)
      forces = self._peak_forces[active_ids] * envelope.view(-1, 1, 1)
      torques = torch.zeros_like(forces)
      self._asset.write_external_wrench_to_sim(
        forces, torques, env_ids=active_ids, body_ids=self._body_ids
      )

    inactive = (~self._active) & step_mask
    self._cooldown_remaining[inactive] -= dt
    trigger = inactive & (self._cooldown_remaining <= 0.0)
    if not trigger.any():
      return

    trigger_ids = trigger.nonzero(as_tuple=False).squeeze(-1)
    n = int(trigger_ids.numel())
    directions = torch.randn(n, self._num_bodies, 3, device=self._device)
    if horizontal_only:
      directions[..., 2] = 0.0
    directions = directions / torch.norm(directions, dim=-1, keepdim=True).clamp_min(
      1.0e-6
    )
    magnitudes = self._sample_range(force_range, (n, self._num_bodies, 1))
    forces = directions * magnitudes
    torques = torch.zeros_like(forces)

    duration = self._sample_range(duration_s, (n,))
    self._peak_forces[trigger_ids] = forces
    self._duration[trigger_ids] = duration
    self._time_remaining[trigger_ids] = duration
    self._asset.write_external_wrench_to_sim(
      torch.zeros_like(forces), torques, env_ids=trigger_ids, body_ids=self._body_ids
    )
    self._active[trigger_ids] = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    reset_ids = self._resolve_env_ids(env_ids)
    if reset_ids.numel() == 0:
      return

    active_ids = reset_ids[self._active[reset_ids]]
    if active_ids.numel() > 0:
      zeros = torch.zeros(
        (int(active_ids.numel()), self._num_bodies, 3), device=self._device
      )
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=active_ids, body_ids=self._body_ids
      )

    n = int(reset_ids.numel())
    self._active[reset_ids] = False
    self._time_remaining[reset_ids] = 0.0
    self._duration[reset_ids] = 1.0
    self._peak_forces[reset_ids] = 0.0
    self._cooldown_remaining[reset_ids] = self._sample_range(self._cooldown_s, (n,))

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._active.any():
      return
    viz = self._viz_cfg
    wrench = self._asset.data.body_external_wrench
    com_pos = self._asset.data.body_com_pos_w
    body_ids = self._selected_body_ids.detach().cpu().tolist()
    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not self._active[env_idx]:
        continue
      for body_id in body_ids:
        force = wrench[env_idx, body_id, :3]
        if torch.norm(force).item() < viz.min_force:
          continue
        start = com_pos[env_idx, body_id].cpu().numpy()
        end = start + force.cpu().numpy() * viz.scale
        visualizer.add_arrow(start=start, end=end, color=viz.rgba, width=viz.width)

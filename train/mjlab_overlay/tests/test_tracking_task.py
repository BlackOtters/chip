"""Tests specific to motion tracking tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  G1_CHIP_UNDESIRED_CONTACT_BODY_NAMES,
)
from mjlab.tasks.tracking.mdp import ChipMotionCommandCfg, MotionCommandCfg


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-Unitree-G1-CHIP",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-Unitree-G1-CHIP",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_chip_tracking_task_has_compliance_terms() -> None:
  """CHIP tracking task should expose compliance inputs and train-time impulses."""
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-CHIP")
  play_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-CHIP", play=True)

  assert isinstance(cfg.commands["motion"], ChipMotionCommandCfg)
  assert "chip_eef_impulse" in cfg.events
  assert "chip_eef_impulse" not in play_cfg.events

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  assert "chip_compliance" in actor_terms
  assert "chip_reference_keypoint_pos_b" in actor_terms
  assert "chip_reference_keypoint_ori_b" in actor_terms
  assert "chip_eef_force_b" not in actor_terms

  assert "chip_compliance" in critic_terms
  assert "chip_reference_keypoint_pos_b" in critic_terms
  assert "chip_reference_keypoint_ori_b" in critic_terms
  assert "chip_eef_force_b" in critic_terms

  for term_name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
    assert actor_terms[term_name].history_length == 10
    assert critic_terms[term_name].history_length == 10

  assert cfg.scene.sensors is not None
  undesired_sensor = next(s for s in cfg.scene.sensors if s.name == "undesired_contact")
  assert undesired_sensor.primary.pattern == r".*"
  assert undesired_sensor.primary.exclude == G1_CHIP_UNDESIRED_CONTACT_BODY_NAMES

  assert "force_estimator" not in cfg.observations
  assert "force_target" not in cfg.observations

  assert "chip_eef_hindsight_pos" not in cfg.rewards
  assert "motion_global_root_pos" not in cfg.rewards
  assert "self_collisions" not in cfg.rewards
  assert "motion_3pt_pos_rel" in cfg.rewards
  assert "motion_3pt_ori_rel" in cfg.rewards
  assert "undesired_contacts" in cfg.rewards
  assert cfg.rewards["motion_body_pos"].func == mdp.motion_relative_body_position_error_exp

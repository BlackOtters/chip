"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import ChipMotionCommandCfg, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


G1_TRACKING_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

G1_CHIP_EEF_BODY_NAMES = (
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

G1_CHIP_UNDESIRED_CONTACT_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

G1_CHIP_COMMAND_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)

CHIP_HISTORY_LENGTH = 10


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = G1_TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_flat_chip_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 CHIP-style compliant tracking configuration."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)

  undesired_contact_cfg = ContactSensorCfg(
    name="undesired_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r".*",
      entity="robot",
      exclude=G1_CHIP_UNDESIRED_CONTACT_BODY_NAMES,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (undesired_contact_cfg,)

  base_motion_cmd = cfg.commands["motion"]
  assert isinstance(base_motion_cmd, MotionCommandCfg)
  cfg.commands["motion"] = ChipMotionCommandCfg(
    resampling_time_range=base_motion_cmd.resampling_time_range,
    debug_vis=base_motion_cmd.debug_vis,
    entity_name=base_motion_cmd.entity_name,
    motion_file=base_motion_cmd.motion_file,
    anchor_body_name=base_motion_cmd.anchor_body_name,
    body_names=base_motion_cmd.body_names,
    eef_body_names=G1_CHIP_EEF_BODY_NAMES,
    command_joint_names=G1_CHIP_COMMAND_JOINT_NAMES,
    pose_range=base_motion_cmd.pose_range,
    velocity_range=base_motion_cmd.velocity_range,
    joint_position_range=base_motion_cmd.joint_position_range,
    adaptive_kernel_size=base_motion_cmd.adaptive_kernel_size,
    adaptive_lambda=base_motion_cmd.adaptive_lambda,
    adaptive_uniform_ratio=base_motion_cmd.adaptive_uniform_ratio,
    adaptive_alpha=base_motion_cmd.adaptive_alpha,
    sampling_mode=base_motion_cmd.sampling_mode,
  )

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["chip_compliance"] = ObservationTermCfg(
      func=mdp.chip_compliance,
      params={"command_name": "motion"},
    )
    cfg.observations[group_name].terms["chip_reference_keypoint_pos_b"] = (
      ObservationTermCfg(
        func=mdp.chip_reference_keypoint_pos_b,
        params={"command_name": "motion"},
      )
    )
    cfg.observations[group_name].terms["chip_reference_keypoint_ori_b"] = (
      ObservationTermCfg(
        func=mdp.chip_reference_keypoint_ori_b,
        params={"command_name": "motion"},
      )
    )

  # Actor does not observe privileged external force directly.
  cfg.observations["actor"].terms.pop("chip_eef_force_b", None)
  cfg.observations["critic"].terms["chip_eef_force_b"] = ObservationTermCfg(
    func=mdp.chip_eef_force_b,
    params={"command_name": "motion"},
  )

  # Feed 10-step proprioceptive/action history directly to actor and critic.
  for group_name in ("actor", "critic"):
    for term_name in (
      "base_lin_vel",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ):
      cfg.observations[group_name].terms[term_name].history_length = CHIP_HISTORY_LENGTH
      cfg.observations[group_name].terms[term_name].flatten_history_dim = True

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  # CHIP reward structure:
  # - keep relative tracking rewards
  # - remove global tracking-only rewards
  # - add 3-point relative tracking rewards
  # - keep action/joint penalties and switch contact penalty to undesired contacts
  cfg.rewards.pop("motion_global_root_pos", None)
  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["motion_3pt_pos_rel"] = cfg.rewards["motion_body_pos"].__class__(
    func=mdp.motion_relative_body_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.3, "body_names": G1_CHIP_EEF_BODY_NAMES},
  )
  cfg.rewards["motion_3pt_ori_rel"] = cfg.rewards["motion_body_ori"].__class__(
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4, "body_names": G1_CHIP_EEF_BODY_NAMES},
  )
  cfg.rewards["undesired_contacts"] = cfg.rewards["action_rate_l2"].__class__(
    func=mdp.undesired_contacts_cost,
    weight=-0.1,
    params={"sensor_name": "undesired_contact", "force_threshold": 1.0},
  )

  if not play:
    cfg.events["chip_eef_impulse"] = EventTermCfg(
      mode="step",
      func=mdp.apply_chip_eef_impulse,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=G1_CHIP_EEF_BODY_NAMES,
          preserve_order=True,
        ),
        "force_range": (0.0, 40.0),
        "duration_s": (1.0, 3.0),
        "cooldown_s": (1.0, 4.0),
      },
    )

  return cfg

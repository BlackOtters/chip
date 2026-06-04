from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)
from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_chip_policy_cfg import G1ChipTrackingPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401

# ======================== Custom Configs ======================== #
"""
Add your custom config here.
"""

_G1_CHIP_JOINT_NAMES = (
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_G1_CHIP_KP = {
    "left_hip_pitch_joint": 40.17923863450712,
    "left_hip_roll_joint": 99.09842777666111,
    "left_hip_yaw_joint": 40.17923863450712,
    "left_knee_joint": 99.09842777666111,
    "left_ankle_pitch_joint": 28.50124619574858,
    "left_ankle_roll_joint": 28.50124619574858,
    "right_hip_pitch_joint": 40.17923863450712,
    "right_hip_roll_joint": 99.09842777666111,
    "right_hip_yaw_joint": 40.17923863450712,
    "right_knee_joint": 99.09842777666111,
    "right_ankle_pitch_joint": 28.50124619574858,
    "right_ankle_roll_joint": 28.50124619574858,
    "waist_yaw_joint": 40.17923863450712,
    "waist_roll_joint": 28.50124619574858,
    "waist_pitch_joint": 28.50124619574858,
    "left_shoulder_pitch_joint": 14.25062309787429,
    "left_shoulder_roll_joint": 14.25062309787429,
    "left_shoulder_yaw_joint": 14.25062309787429,
    "left_elbow_joint": 14.25062309787429,
    "left_wrist_roll_joint": 14.25062309787429,
    "left_wrist_pitch_joint": 16.77832748089279,
    "left_wrist_yaw_joint": 16.77832748089279,
    "right_shoulder_pitch_joint": 14.25062309787429,
    "right_shoulder_roll_joint": 14.25062309787429,
    "right_shoulder_yaw_joint": 14.25062309787429,
    "right_elbow_joint": 14.25062309787429,
    "right_wrist_roll_joint": 14.25062309787429,
    "right_wrist_pitch_joint": 16.77832748089279,
    "right_wrist_yaw_joint": 16.77832748089279,
}

_G1_CHIP_KD = {
    "left_hip_pitch_joint": 2.5578897753402656,
    "left_hip_roll_joint": 6.30880185331632,
    "left_hip_yaw_joint": 2.5578897753402656,
    "left_knee_joint": 6.30880185331632,
    "left_ankle_pitch_joint": 1.8144456865329854,
    "left_ankle_roll_joint": 1.8144456865329854,
    "right_hip_pitch_joint": 2.5578897753402656,
    "right_hip_roll_joint": 6.30880185331632,
    "right_hip_yaw_joint": 2.5578897753402656,
    "right_knee_joint": 6.30880185331632,
    "right_ankle_pitch_joint": 1.8144456865329854,
    "right_ankle_roll_joint": 1.8144456865329854,
    "waist_yaw_joint": 2.5578897753402656,
    "waist_roll_joint": 1.8144456865329854,
    "waist_pitch_joint": 1.8144456865329854,
    "left_shoulder_pitch_joint": 0.9072228432664927,
    "left_shoulder_roll_joint": 0.9072228432664927,
    "left_shoulder_yaw_joint": 0.9072228432664927,
    "left_elbow_joint": 0.9072228432664927,
    "left_wrist_roll_joint": 0.9072228432664927,
    "left_wrist_pitch_joint": 1.0681415021594705,
    "left_wrist_yaw_joint": 1.0681415021594705,
    "right_shoulder_pitch_joint": 0.9072228432664927,
    "right_shoulder_roll_joint": 0.9072228432664927,
    "right_shoulder_yaw_joint": 0.9072228432664927,
    "right_elbow_joint": 0.9072228432664927,
    "right_wrist_roll_joint": 0.9072228432664927,
    "right_wrist_pitch_joint": 1.0681415021594705,
    "right_wrist_yaw_joint": 1.0681415021594705,
}

_G1_CHIP_EFFORT = {
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 50.0,
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 50.0,
    "right_ankle_roll_joint": 50.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}

@cfg_registry.register
class g1_dev(RlPipelineCfg):
    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()


@cfg_registry.register
class g1_chip(RlPipelineCfg):
    """CHIP tracking policy on MuJoCo with mjlab-aligned sim params."""

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=False,
        sim_dt=0.005,
        sim_decimation=4,
        sim_iterations=10,
        sim_ls_iterations=20,
        actuator_mode="position_servo",
        actuator_kp_override=_G1_CHIP_KP,
        actuator_kd_override=_G1_CHIP_KD,
        actuator_effort_override=_G1_CHIP_EFFORT,
        joint_armature_override={
            "left_hip_pitch_joint": 0.0101775200413223,
            "left_hip_roll_joint": 0.025101925,
            "left_hip_yaw_joint": 0.0101775200413223,
            "left_knee_joint": 0.025101925,
            "left_ankle_pitch_joint": 0.00721945,
            "left_ankle_roll_joint": 0.00721945,
            "right_hip_pitch_joint": 0.0101775200413223,
            "right_hip_roll_joint": 0.025101925,
            "right_hip_yaw_joint": 0.0101775200413223,
            "right_knee_joint": 0.025101925,
            "right_ankle_pitch_joint": 0.00721945,
            "right_ankle_roll_joint": 0.00721945,
            "waist_yaw_joint": 0.0101775200413223,
            "waist_roll_joint": 0.00721945,
            "waist_pitch_joint": 0.00721945,
            "left_shoulder_pitch_joint": 0.003609725,
            "left_shoulder_roll_joint": 0.003609725,
            "left_shoulder_yaw_joint": 0.003609725,
            "left_elbow_joint": 0.003609725,
            "left_wrist_roll_joint": 0.003609725,
            "left_wrist_pitch_joint": 0.00425,
            "left_wrist_yaw_joint": 0.00425,
            "right_shoulder_pitch_joint": 0.003609725,
            "right_shoulder_roll_joint": 0.003609725,
            "right_shoulder_yaw_joint": 0.003609725,
            "right_elbow_joint": 0.003609725,
            "right_wrist_roll_joint": 0.003609725,
            "right_wrist_pitch_joint": 0.00425,
            "right_wrist_yaw_joint": 0.00425,
        },
        # Damping is already injected by CHIP controller torque law:
        # tau = kp*(q_ref-q) - kd*dq. Keep passive joint damping at zero.
        joint_damping_override={name: 0.0 for name in _G1_CHIP_JOINT_NAMES},
        joint_frictionloss_override={name: 0.0 for name in _G1_CHIP_JOINT_NAMES},
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "r": "[MOTION_RESET]",
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "<": "[MOTION_FADE_IN]",
                ">": "[MOTION_FADE_OUT]",
            }
        ),
    ]
    policy: G1ChipTrackingPolicyCfg = G1ChipTrackingPolicyCfg(
        motion_time_start=0.0,
        motion_time_end=21.2,
        compliance_range=[0.15, 0.15],
    )


@cfg_registry.register
class g1_chip_real(g1_chip):
    """CHIP tracking policy on real G1 hardware."""

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
        born_place_align=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]
    do_safety_check: bool = True

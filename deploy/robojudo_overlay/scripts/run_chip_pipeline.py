"""Run CHIP policy in RoboJuDo.

Extends run_pipeline with CHIP-specific path overrides:
- --checkpoint-file
- --motion-file

Example:
  python scripts/run_chip_pipeline.py -c g1_chip \
      --checkpoint-file /abs/path/model_29999.pt \
      --motion-file /abs/path/walk_xxx.npz
"""

# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import numpy as np

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CHIP policy pipeline")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1_chip",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to CHIP RSL-RL checkpoint (.pt), overrides config",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Path to CHIP motion file (.npz), overrides config",
    )
    parser.add_argument(
        "--motion-time-start",
        type=float,
        default=None,
        help="Motion start time in seconds, overrides config",
    )
    parser.add_argument(
        "--motion-time-end",
        type=float,
        default=None,
        help="Motion end time in seconds, overrides config",
    )
    parser.add_argument(
        "--motion-start-blend-seconds",
        type=float,
        default=None,
        help="Smooth startup blend duration on MOTION_RESET (seconds), overrides config",
    )
    parser.add_argument(
        "--prepare-seconds",
        type=float,
        default=None,
        help="Ramp/blend duration for prepare(); defaults to pipeline values",
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Skip prepare and only hold default pose in sim",
    )
    parser.add_argument(
        "--start-mode",
        type=str,
        choices=["mjlab", "safe"],
        default="mjlab",
        help="Startup strategy: mjlab=reborn to motion frame and run; safe=prepare then press R",
    )
    parser.add_argument(
        "--mouse-force-drag",
        action="store_true",
        help="Enable mouse drag perturbation in MuJoCo sim for body-force testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)
    cfg: RlPipelineCfg = config_manager.get_cfg()

    if args.checkpoint_file is not None and hasattr(cfg, "policy"):
        cfg.policy.checkpoint_file = args.checkpoint_file
    if args.motion_file is not None and hasattr(cfg, "policy"):
        cfg.policy.motion_file = args.motion_file
    if args.motion_time_start is not None and hasattr(cfg, "policy"):
        cfg.policy.motion_time_start = args.motion_time_start
    if args.motion_time_end is not None and hasattr(cfg, "policy"):
        cfg.policy.motion_time_end = args.motion_time_end
    if args.motion_start_blend_seconds is not None and hasattr(cfg, "policy"):
        cfg.policy.motion_start_blend_seconds = args.motion_start_blend_seconds

    pipeline_type = cfg.pipeline_type
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")
    pipeline = pipeline_class(cfg=cfg)

    if args.mouse_force_drag and hasattr(pipeline.env, "enable_mouse_force_drag"):
        pipeline.env.enable_mouse_force_drag(True)  # type: ignore[attr-defined]

    if args.start_mode == "mjlab":
        if hasattr(pipeline.policy, "get_motion_init_qpos_xyzw") and hasattr(pipeline.env, "reborn"):
            try:
                motion_init = pipeline.policy.get_motion_init_qpos_xyzw()
                motion_qvel = (
                    pipeline.policy.get_motion_init_qvel()
                    if hasattr(pipeline.policy, "get_motion_init_qvel")
                    else None
                )
                pipeline.env.reborn(init_qpos=motion_init, init_qvel=motion_qvel)
                if pipeline._has_default_pose_mode:
                    pipeline._set_default_pose_mode(False)
            except Exception as exc:
                logger.warning(f"Motion-frame reborn failed, fallback to safe mode: {exc}")
                args.start_mode = "safe"
        else:
            args.start_mode = "safe"

    if args.start_mode == "safe":
        if args.no_prepare:
            if pipeline._has_default_pose_mode:
                pipeline._set_default_pose_mode(True)
                logger.warning("Holding default pose, press R to start motion")
        else:
            init_motor_angle = np.asarray(pipeline.policy.get_init_dof_pos(), dtype=np.float32)
            pipeline.prepare(
                init_motor_angle=init_motor_angle,
                prepare_seconds=args.prepare_seconds,
            )

    while True:
        time_start = time.time()
        pipeline.step()
        time_end = time.time()
        time_diff = time_end - time_start

        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()

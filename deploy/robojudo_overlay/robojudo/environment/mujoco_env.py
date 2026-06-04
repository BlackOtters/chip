import logging
import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split(".")))

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation
        self.actuator_mode = cfg_env.actuator_mode

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        if cfg_env.sim_iterations is not None:
            self.model.opt.iterations = int(cfg_env.sim_iterations)
        if cfg_env.sim_ls_iterations is not None:
            self.model.opt.ls_iterations = int(cfg_env.sim_ls_iterations)

        self._apply_joint_dynamics_overrides(cfg_env)
        self._apply_actuator_overrides(cfg_env)

        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        self._torso_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self._torso_name,
        )
        self._imu_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "imu_in_pelvis",
        )
        self._imu_site_body_id = (
            int(self.model.site_bodyid[self._imu_site_id]) if self._imu_site_id >= 0 else -1
        )
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading
        self._mouse_force_drag_enabled = False
        self._mouse_force_drag_selected_body = -1
        self._viewer_mouse_button_callback = self.viewer._mouse_button_callback

        self._apply_random_heading()

        self.update()  # get initial state

    def enable_mouse_force_drag(self, enabled: bool = True):
        """Enable or disable mouse drag perturbation for MuJoCo simulation."""
        self._mouse_force_drag_enabled = bool(enabled)
        if not hasattr(self, "viewer") or self.viewer is None:
            return

        if enabled:
            self._viewer_mouse_button_callback = self.viewer._mouse_button_callback
            glfw.set_mouse_button_callback(self.viewer.window, self._mouse_force_mouse_button_callback)
            logger.info(
                "[MujocoEnv] mouse force drag enabled: left click selects a body, "
                "left drag rotates it, right drag translates it"
            )
        else:
            glfw.set_mouse_button_callback(self.viewer.window, self.viewer._mouse_button_callback)
            self._mouse_force_drag_selected_body = -1
            logger.info("[MujocoEnv] mouse force drag disabled")

    def _pick_body_at_cursor(self, window) -> tuple[int, np.ndarray, int]:
        """Pick body under the current cursor position."""
        x, y = glfw.get_cursor_pos(window)
        self.viewer._last_mouse_x = int(self.viewer._scale * x)
        self.viewer._last_mouse_y = int(self.viewer._scale * y)

        width, height = self.viewer.viewport.width, self.viewer.viewport.height
        if width <= 0 or height <= 0:
            return -1, np.zeros(3, dtype=np.float64), -1

        aspectratio = width / height
        relx = x / width
        rely = (height - y) / height
        selpnt = np.zeros((3, 1), dtype=np.float64)
        selgeom = np.zeros((1, 1), dtype=np.int32)
        selflex = np.zeros((1, 1), dtype=np.int32)
        selskin = np.zeros((1, 1), dtype=np.int32)

        if MUJOCO_VERSION >= (3, 0, 0):
            selbody = mujoco.mjv_select(
                self.model,
                self.data,
                self.viewer.vopt,
                aspectratio,
                relx,
                rely,
                self.viewer.scn,
                selpnt,
                selgeom,
                selflex,
                selskin,
            )
        else:
            selbody = mujoco.mjv_select(
                self.model,
                self.data,
                self.viewer.vopt,
                aspectratio,
                relx,
                rely,
                self.viewer.scn,
                selpnt,
                selgeom,
                selskin,
            )

        skinselect = int(selskin[0, 0]) if selskin.size > 0 else -1
        return int(selbody), selpnt.flatten(), skinselect

    def _mouse_force_mouse_button_callback(self, window, button, act, mods):
        """Mouse callback that turns clicks into body perturbations."""
        if not self._mouse_force_drag_enabled:
            return self._viewer_mouse_button_callback(window, button, act, mods)

        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            return self._viewer_mouse_button_callback(window, button, act, mods)

        x, y = glfw.get_cursor_pos(window)
        self.viewer._last_mouse_x = int(self.viewer._scale * x)
        self.viewer._last_mouse_y = int(self.viewer._scale * y)

        if button == glfw.MOUSE_BUTTON_LEFT:
            if act == glfw.PRESS:
                self.viewer._button_left_pressed = True
                selbody, selpnt, skinselect = self._pick_body_at_cursor(window)
                if selbody > 0:
                    self._mouse_force_drag_selected_body = selbody
                    self.viewer.pert.select = selbody
                    self.viewer.pert.skinselect = skinselect
                    self.viewer.cam.lookat = selpnt
                    vec = selpnt - self.data.xpos[selbody]
                    self.viewer.pert.localpos = self.data.xmat[selbody].reshape(3, 3).dot(vec)
                    if not self.viewer.pert.active:
                        mujoco.mjv_initPerturb(self.model, self.data, self.viewer.scn, self.viewer.pert)
                    self.viewer.pert.active = mujoco.mjtPertBit.mjPERT_ROTATE
                    return

                self._mouse_force_drag_selected_body = -1
                self.viewer._button_left_pressed = False
                return self._viewer_mouse_button_callback(window, button, act, mods)

            if act == glfw.RELEASE:
                self.viewer._button_left_pressed = False
                if not self.viewer._button_right_pressed:
                    self.viewer.pert.active = 0
                return

        if button == glfw.MOUSE_BUTTON_RIGHT:
            if act == glfw.PRESS:
                if self.viewer.pert.select > 0:
                    self.viewer._button_right_pressed = True
                    self.viewer.pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                    return
                return self._viewer_mouse_button_callback(window, button, act, mods)

            if act == glfw.RELEASE:
                if self.viewer.pert.select > 0:
                    self.viewer._button_right_pressed = False
                    if not self.viewer._button_left_pressed:
                        self.viewer.pert.active = 0
                    return
                return self._viewer_mouse_button_callback(window, button, act, mods)

        return self._viewer_mouse_button_callback(window, button, act, mods)

    def _apply_joint_dynamics_overrides(self, cfg_env: MujocoEnvCfg):
        """Apply optional per-joint MuJoCo dynamics overrides from config."""
        armature_map = cfg_env.joint_armature_override or {}
        damping_map = cfg_env.joint_damping_override or {}
        friction_map = cfg_env.joint_frictionloss_override or {}
        if not armature_map and not damping_map and not friction_map:
            return

        for joint_name, value in armature_map.items():
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id < 0:
                logger.warning("[MujocoEnv] armature override skipped, unknown joint: %s", joint_name)
                continue
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            if dof_adr < 0:
                logger.warning("[MujocoEnv] armature override skipped, invalid dof adr for: %s", joint_name)
                continue
            self.model.dof_armature[dof_adr] = float(value)

        for joint_name, value in damping_map.items():
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id < 0:
                logger.warning("[MujocoEnv] damping override skipped, unknown joint: %s", joint_name)
                continue
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            if dof_adr < 0:
                logger.warning("[MujocoEnv] damping override skipped, invalid dof adr for: %s", joint_name)
                continue
            self.model.dof_damping[dof_adr] = float(value)

        for joint_name, value in friction_map.items():
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id < 0:
                logger.warning("[MujocoEnv] frictionloss override skipped, unknown joint: %s", joint_name)
                continue
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            if dof_adr < 0:
                logger.warning("[MujocoEnv] frictionloss override skipped, invalid dof adr for: %s", joint_name)
                continue
            self.model.dof_frictionloss[dof_adr] = float(value)

    def _apply_actuator_overrides(self, cfg_env: MujocoEnvCfg):
        """Apply optional actuator overrides from config."""
        if cfg_env.actuator_mode != "position_servo":
            return

        kp_map = cfg_env.actuator_kp_override or {}
        kd_map = cfg_env.actuator_kd_override or {}
        effort_map = cfg_env.actuator_effort_override or {}
        if not kp_map:
            logger.warning("[MujocoEnv] position_servo mode enabled but actuator_kp_override is empty")
            return

        for aid in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if actuator_name is None or actuator_name not in kp_map:
                continue

            kp = float(kp_map[actuator_name])
            kd = float(kd_map.get(actuator_name, 0.0))
            effort = effort_map.get(actuator_name, None)

            # Match MuJoCo position servo force form:
            # force = kp * ctrl + (-kp) * q + (-kd) * qdot
            self.model.actuator_dyntype[aid] = mujoco.mjtDyn.mjDYN_NONE
            self.model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE

            self.model.actuator_gainprm[aid, :] = 0.0
            self.model.actuator_biasprm[aid, :] = 0.0
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kd

            self.model.actuator_ctrllimited[aid] = 0
            self.model.actuator_ctrlrange[aid, 0] = -1.0e6
            self.model.actuator_ctrlrange[aid, 1] = 1.0e6

            if effort is not None:
                self.model.actuator_forcelimited[aid] = 1
                e = float(effort)
                self.model.actuator_forcerange[aid, 0] = -e
                self.model.actuator_forcerange[aid, 1] = e

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None, init_qvel=None):
        if init_qpos is not None:
            init_qpos = np.asarray(init_qpos, dtype=np.float32)
            root_quat_xyzw = init_qpos[3:7].astype(np.float32)
            self.data.qpos[0:3] = init_qpos[0:3]
            # RoboJuDo policies use scipy/xyzw quaternions; MuJoCo qpos uses wxyz.
            self.data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]
            self.data.qpos[7 : 7 + self.num_dofs] = init_qpos[7 : 7 + self.num_dofs]
            if init_qvel is None:
                self.data.qvel[:] = 0.0
            else:
                init_qvel = np.asarray(init_qvel, dtype=np.float32)
                self.data.qvel[:] = 0.0
                self.data.qvel[0:3] = init_qvel[0:3]

                # MuJoCo stores free-joint angular velocity in body frame.
                # The motion npz stores world-frame angular velocity, so convert
                # before writing qvel to match mjlab's write_root_velocity().
                root_quat_norm = float(np.linalg.norm(root_quat_xyzw))
                if root_quat_norm > 1e-8:
                    root_quat_xyzw = root_quat_xyzw / root_quat_norm
                    root_ang_vel_b = quat_rotate_inverse_np(root_quat_xyzw, init_qvel[3:6])
                else:
                    root_ang_vel_b = init_qvel[3:6]
                self.data.qvel[3:6] = root_ang_vel_b.astype(np.float32)

                qvel_joint_len = min(self.num_dofs, max(init_qvel.shape[0] - 6, 0))
                if qvel_joint_len > 0:
                    self.data.qvel[6 : 6 + qvel_joint_len] = init_qvel[6 : 6 + qvel_joint_len]
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        root_lin_vel_w = self.data.qvel.astype(np.float32)[0:3]

        if self._imu_site_id >= 0 and self._imu_site_body_id >= 0:
            body_ang_vel_b = self.data.qvel.astype(np.float32)[3:6]
            body_ang_vel_w = self.data.xmat[self._imu_site_body_id].reshape(3, 3) @ body_ang_vel_b
            imu_offset_w = self.data.site_xpos[self._imu_site_id] - self.data.xpos[self._imu_site_body_id]
            imu_lin_vel_w = root_lin_vel_w + np.cross(body_ang_vel_w, imu_offset_w)
            site_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
            self._imu_lin_vel = (site_xmat.T @ imu_lin_vel_w).astype(np.float32)
        else:
            self._imu_lin_vel = None

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)
        lin_vel = quat_rotate_inverse_np(quat, root_lin_vel_w).astype(np.float32)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

        if self._torso_body_id >= 0:
            self._torso_pos = self.data.xpos[self._torso_body_id].astype(np.float32).copy()
            # MuJoCo xquat is wxyz; RoboJuDo env data uses xyzw.
            self._torso_quat = self.data.xquat[self._torso_body_id].astype(np.float32)[[1, 2, 3, 0]].copy()

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            if self.actuator_mode == "position_servo":
                self.data.ctrl[:] = pd_target
            else:
                torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)

import logging
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import ChipTrackingPolicyCfg
from robojudo.utils.util_func import matrix_from_quat, quat_rotate_inverse_np

logger = logging.getLogger(__name__)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_to_mat_first_two_cols(q_xyzw: np.ndarray) -> np.ndarray:
    m = matrix_from_quat(q_xyzw)
    # Match PyTorch flatten(mat[..., :2]) order.
    return np.asarray([m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 0], m[2, 1]], dtype=np.float32)


class _RslRlActor(torch.nn.Module):
    """Minimal RSL-RL actor for inference from actor_state_dict."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        dims = (in_dim, *hidden_dims, out_dim)
        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class _SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class _ChipTemporalCrossAttentionCore(torch.nn.Module):
    """Inference-only CHIP attention encoder matching mjlab checkpoints."""

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
    ):
        super().__init__()
        self.history_length = history_length
        self.prefix_dim = 30 + 3 + 6
        self.history_step_dims = (3, 3, 29, 29, 29)
        self.history_step_dim = sum(self.history_step_dims)
        self.history_flat_dims = tuple(d * history_length for d in self.history_step_dims)
        self.history_flat_dim = sum(self.history_flat_dims)
        self.tail_dim = 3 + 9 + 18
        self.base_obs_dim = self.prefix_dim + self.history_flat_dim + self.tail_dim
        if obs_dim != self.base_obs_dim:
            raise ValueError(f"CHIP attention actor expects obs_dim={self.base_obs_dim}, got {obs_dim}")

        context_dim = self.prefix_dim + self.tail_dim
        self.history_proj = torch.nn.Linear(self.history_step_dim, embed_dim)
        self.pos_encoding = _SinusoidalPositionalEncoding(embed_dim, history_length)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_proj = torch.nn.Sequential(
            torch.nn.Linear(context_dim, embed_dim),
            torch.nn.ELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.fusion_norm = torch.nn.LayerNorm(embed_dim)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, embed_dim),
            torch.nn.ELU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ELU(),
        )

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
        context = torch.cat((prefix, tail), dim=-1)
        return history, context

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        history, context = self._split_flat(flat_obs)
        history_tokens = self.history_proj(history)
        history_tokens = self.pos_encoding(history_tokens)
        history_tokens = self.history_encoder(history_tokens)

        context_token = self.context_proj(context)
        attended, _ = self.cross_attention(
            context_token.unsqueeze(1),
            history_tokens,
            history_tokens,
            need_weights=False,
        )
        attended = attended.squeeze(1)
        fused = self.fusion_norm(context_token + attended)
        pooled = history_tokens.mean(dim=1)
        recent = history_tokens[:, -1]
        return self.fusion(torch.cat((fused, recent, pooled), dim=-1))


class _ChipAttentionActor(torch.nn.Module):
    """Inference-only actor for mjlab ChipTemporalCrossAttentionModel checkpoints."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...],
        out_dim: int,
        history_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
    ):
        super().__init__()
        self.attention_core = _ChipTemporalCrossAttentionCore(
            obs_dim=obs_dim,
            history_length=history_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
        )
        dims = (embed_dim, *hidden_dims, out_dim)
        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attention_core(obs))


class _ChipMotionLoader:
    """NPZ motion loader using mjlab-style discrete frame indexing."""

    def __init__(self, motion_path: str, loop: bool = True):
        data = np.load(motion_path)
        required = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
        for key in required:
            if key not in data:
                raise ValueError(f"Motion npz missing key: {key}")

        fps = float(np.asarray(data["fps"]).item())
        if fps <= 0.0:
            raise ValueError("motion fps must be > 0")
        self.dt = 1.0 / fps

        self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        self.body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        self.body_quat_wxyz = np.asarray(data["body_quat_w"], dtype=np.float32)
        self.body_lin_vel_w = (
            np.asarray(data["body_lin_vel_w"], dtype=np.float32)
            if "body_lin_vel_w" in data
            else np.zeros_like(self.body_pos_w)
        )
        self.body_ang_vel_w = (
            np.asarray(data["body_ang_vel_w"], dtype=np.float32)
            if "body_ang_vel_w" in data
            else np.zeros_like(self.body_pos_w)
        )

        if self.body_quat_wxyz.shape[-1] != 4:
            raise ValueError("body_quat_w expected [...,4]")

        self.num_frames = int(self.joint_pos.shape[0])
        if self.num_frames < 1:
            raise ValueError("empty motion")
        self.duration = float(self.num_frames * self.dt)
        self.loop = loop

        self._idx0 = 0
        self._idx1 = 0
        self._blend = 0.0
        self._pos_offset = np.zeros(3, dtype=np.float32)

    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)

    def reset(
        self,
        root_pos_w: np.ndarray,
        t: float = 0.0,
        time_start: float = 0.0,
        time_end: float | None = None,
    ):
        self.update(t, time_start=time_start, time_end=time_end)
        # Use un-offset reference root so repeated resets do not accumulate drift.
        root_ref = self.body_pos_w[self._idx0, 0].astype(np.float32)
        self._pos_offset = np.asarray(root_pos_w, dtype=np.float32) - root_ref
        self._pos_offset[2] = 0.0

    def update(self, time_sec: float, time_start: float = 0.0, time_end: float | None = None):
        if self.num_frames <= 1:
            self._idx0 = 0
            self._idx1 = 0
            self._blend = 0.0
            return

        start = float(np.clip(time_start, 0.0, self.duration))
        end_raw = self.duration if time_end is None else float(time_end)
        end = float(np.clip(end_raw, start, self.duration))

        t = float(time_sec)
        if time_end is None and self.loop:
            t = start + t
            t = float(np.fmod(t, self.duration))
            if t < 0.0:
                t += self.duration
        else:
            elapsed = float(np.clip(t, 0.0, max(end - start, 0.0)))
            t = start + elapsed

        frame = int(np.floor(t / self.dt + 1e-6)) if self.dt > 0.0 else 0
        self._idx0 = int(np.clip(frame, 0, self.num_frames - 1))
        self._idx1 = self._idx0
        self._blend = 0.0

    def _lerp(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a

    def _slerp_xyzw(self, qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot

        if dot > 0.9995:
            out = qa + self._blend * (qb - qa)
            n = np.linalg.norm(out)
            return (out / max(n, 1e-8)).astype(np.float32)

        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * self._blend
        sin_theta = np.sin(theta)
        s0 = np.sin(theta_0 - theta) / max(sin_theta_0, 1e-8)
        s1 = sin_theta / max(sin_theta_0, 1e-8)
        return (s0 * qa + s1 * qb).astype(np.float32)

    def joint_pos_at(self) -> np.ndarray:
        return self._lerp(self.joint_pos[self._idx0], self.joint_pos[self._idx1]).astype(np.float32)

    def joint_vel_at(self) -> np.ndarray:
        return self._lerp(self.joint_vel[self._idx0], self.joint_vel[self._idx1]).astype(np.float32)

    def raw_body_pos(self, body_id: int) -> np.ndarray:
        return self.body_pos_w[self._idx0, body_id].astype(np.float32)

    def body_pos(self, body_id: int) -> np.ndarray:
        return self.raw_body_pos(body_id) + self._pos_offset

    def body_lin_vel(self, body_id: int) -> np.ndarray:
        return self.body_lin_vel_w[self._idx0, body_id].astype(np.float32)

    def body_ang_vel(self, body_id: int) -> np.ndarray:
        return self.body_ang_vel_w[self._idx0, body_id].astype(np.float32)

    def body_quat(self, body_id: int) -> np.ndarray:
        return self._wxyz_to_xyzw(self.body_quat_wxyz[self._idx0, body_id])


@policy_registry.register
class ChipTrackingPolicy(Policy):
    """CHIP policy using mjlab RSL-RL checkpoint + motion npz."""

    cfg_policy: ChipTrackingPolicyCfg

    def __init__(self, cfg_policy: ChipTrackingPolicyCfg, device: str = "cpu"):
        cfg_policy_updated = cfg_policy.model_copy()
        super().__init__(cfg_policy=cfg_policy_updated, device=device)

        if not cfg_policy.checkpoint_file:
            raise ValueError("ChipTrackingPolicyCfg.checkpoint_file is required")
        if not cfg_policy.motion_file:
            raise ValueError("ChipTrackingPolicyCfg.motion_file is required")

        ckpt_path = Path(cfg_policy.checkpoint_file)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        actor_sd = ckpt.get("actor_state_dict")
        if actor_sd is None:
            raise ValueError("checkpoint missing actor_state_dict")

        self._obs_mean = actor_sd["obs_normalizer._mean"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        self._obs_std = actor_sd["obs_normalizer._std"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        self._obs_std = np.maximum(self._obs_std, 1e-6)

        self._obs_dim = int(self._obs_mean.shape[0])
        self._act_dim = self.num_actions

        if any(k.startswith("attention_core.") for k in actor_sd):
            embed_dim = int(actor_sd["attention_core.history_proj.weight"].shape[0])
            ff_dim = int(actor_sd["attention_core.history_encoder.layers.0.linear1.weight"].shape[0])
            layer_ids = {
                int(k.split(".")[3])
                for k in actor_sd
                if k.startswith("attention_core.history_encoder.layers.")
            }
            num_layers = max(layer_ids) + 1 if layer_ids else 1
            num_heads = 4
            self._actor = _ChipAttentionActor(
                obs_dim=self._obs_dim,
                hidden_dims=(512, 256, 128),
                out_dim=self._act_dim,
                history_length=self.history_length,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
            ).to(self.device)
            self._actor.load_state_dict(
                {k: v for k, v in actor_sd.items() if k.startswith(("attention_core.", "mlp."))},
                strict=True,
            )
            actor_kind = f"attention(embed={embed_dim}, layers={num_layers}, heads={num_heads})"
        else:
            self._actor = _RslRlActor(self._obs_dim, (512, 256, 128), self._act_dim).to(self.device)
            self._actor.load_state_dict(
                {
                    "mlp.0.weight": actor_sd["mlp.0.weight"],
                    "mlp.0.bias": actor_sd["mlp.0.bias"],
                    "mlp.2.weight": actor_sd["mlp.2.weight"],
                    "mlp.2.bias": actor_sd["mlp.2.bias"],
                    "mlp.4.weight": actor_sd["mlp.4.weight"],
                    "mlp.4.bias": actor_sd["mlp.4.bias"],
                    "mlp.6.weight": actor_sd["mlp.6.weight"],
                    "mlp.6.bias": actor_sd["mlp.6.bias"],
                },
                strict=True,
            )
            actor_kind = "mlp"
        self._actor.eval()

        self._motion = _ChipMotionLoader(cfg_policy.motion_file, loop=True)

        joint_names = list(self.cfg_obs_dof.joint_names)
        self._jid = {name: i for i, name in enumerate(joint_names)}
        self._cmd_joint_ids = [self._jid[n] for n in cfg_policy.command_joint_names]

        # Body ids in mjlab motion export convention (world body excluded).
        self._body_ids = {
            "pelvis": 0,
            "torso_link": 15,
            "left_wrist_yaw_link": 22,
            "right_wrist_yaw_link": 29,
        }
        self._key_body_ids = [self._body_ids[n] for n in cfg_policy.keypoint_body_names]

        self._default_dof_pos = np.asarray(self.cfg_action_dof.default_pos, dtype=np.float32)
        self._joint_action_scale = self._build_action_scales(joint_names)

        if cfg_policy.joint_pos_bias is not None:
            if len(cfg_policy.joint_pos_bias) != self.num_dofs:
                raise ValueError(
                    f"joint_pos_bias length {len(cfg_policy.joint_pos_bias)} != {self.num_dofs}"
                )
            self._joint_pos_bias = np.asarray(cfg_policy.joint_pos_bias, dtype=np.float32)
        else:
            self._joint_pos_bias = np.zeros(self.num_dofs, dtype=np.float32)

        self._compliance_range = np.asarray(cfg_policy.compliance_range, dtype=np.float32)
        self._goal_alpha = float(cfg_policy.goal_smoothing_alpha)
        self._rng = np.random.default_rng(seed=42)
        self._motion_time_start = float(max(cfg_policy.motion_time_start, 0.0))
        if cfg_policy.motion_time_end is None:
            self._motion_time_end = float(self._motion.duration)
        else:
            self._motion_time_end = float(
                np.clip(cfg_policy.motion_time_end, self._motion_time_start, self._motion.duration)
            )
        self._start_in_default_pose = bool(cfg_policy.start_in_default_pose)

        logger.info(
            (
                "[ChipTrackingPolicy] loaded checkpoint=%s obs_dim=%d act_dim=%d "
                "actor=%s motion_frames=%d motion_window=[%.3f, %.3f)"
            ),
            ckpt_path,
            self._obs_dim,
            self._act_dim,
            actor_kind,
            self._motion.num_frames,
            self._motion_time_start,
            self._motion_time_end,
        )

        self.reset()

    def _build_default_joint_pos(self, joint_names: list[str], keyframe: dict[str, float]) -> np.ndarray:
        import re

        out = np.zeros(len(joint_names), dtype=np.float32)
        for pattern, value in keyframe.items():
            rex = re.compile(pattern)
            for i, name in enumerate(joint_names):
                if rex.fullmatch(name):
                    out[i] = float(value)
        return out

    def _build_action_scales(self, joint_names: list[str]) -> np.ndarray:
        import re

        patterns = [
            (r".*_elbow_joint", 0.25 * 25.0 / 14.25062309787429),
            (r".*_shoulder_pitch_joint", 0.25 * 25.0 / 14.25062309787429),
            (r".*_shoulder_roll_joint", 0.25 * 25.0 / 14.25062309787429),
            (r".*_shoulder_yaw_joint", 0.25 * 25.0 / 14.25062309787429),
            (r".*_wrist_roll_joint", 0.25 * 25.0 / 14.25062309787429),
            (r".*_hip_pitch_joint", 0.25 * 88.0 / 40.17923863450712),
            (r".*_hip_yaw_joint", 0.25 * 88.0 / 40.17923863450712),
            (r"waist_yaw_joint", 0.25 * 88.0 / 40.17923863450712),
            (r".*_hip_roll_joint", 0.25 * 139.0 / 99.09842777666111),
            (r".*_knee_joint", 0.25 * 139.0 / 99.09842777666111),
            (r".*_wrist_pitch_joint", 0.25 * 5.0 / 16.77832748089279),
            (r".*_wrist_yaw_joint", 0.25 * 5.0 / 16.77832748089279),
            (r"waist_pitch_joint", 0.25 * 50.0 / 28.50124619574858),
            (r"waist_roll_joint", 0.25 * 50.0 / 28.50124619574858),
            (r".*_ankle_pitch_joint", 0.25 * 50.0 / 28.50124619574858),
            (r".*_ankle_roll_joint", 0.25 * 50.0 / 28.50124619574858),
        ]
        scales = np.ones(len(joint_names), dtype=np.float32)
        for i, name in enumerate(joint_names):
            for pattern, value in patterns:
                if re.fullmatch(pattern, name):
                    scales[i] = np.float32(value)
                    break
        return scales

    @staticmethod
    def _normalize_quat(q_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_xyzw, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return q / n

    def _get_anchor_pose(self, env_data, root_pos_w: np.ndarray, root_quat_xyzw: np.ndarray):
        """Anchor pose used by motion_anchor_* observations.

        Matches mjlab tracking setup where anchor body is torso_link.
        Prefer FK torso state from env data; fall back to root pose if unavailable.
        """
        torso_pos = env_data.torso_pos if hasattr(env_data, "torso_pos") else None
        torso_quat = env_data.torso_quat if hasattr(env_data, "torso_quat") else None
        if torso_pos is None or torso_quat is None:
            return root_pos_w, root_quat_xyzw
        return (
            np.asarray(torso_pos, dtype=np.float32),
            self._normalize_quat(np.asarray(torso_quat, dtype=np.float32)),
        )

    def _sample_compliance(self) -> np.ndarray:
        low, high = float(self._compliance_range[0]), float(self._compliance_range[1])
        return self._rng.uniform(low, high, size=(3,)).astype(np.float32)

    def reset(self):
        self._frame = 0
        self._time = 0.0

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        z3 = np.zeros(3, dtype=np.float32)
        z29 = np.zeros(self.num_dofs, dtype=np.float32)

        self._hist_base_lin_vel = deque([z3.copy() for _ in range(self.history_length)], maxlen=self.history_length)
        self._hist_base_ang_vel = deque([z3.copy() for _ in range(self.history_length)], maxlen=self.history_length)
        self._hist_joint_pos_rel = deque([z29.copy() for _ in range(self.history_length)], maxlen=self.history_length)
        self._hist_joint_vel_rel = deque([z29.copy() for _ in range(self.history_length)], maxlen=self.history_length)
        self._hist_action = deque([z29.copy() for _ in range(self.history_length)], maxlen=self.history_length)

        self._chip_goal_initialized = False
        self._eef_goal_pos_w = np.zeros((3, 3), dtype=np.float32)
        self._eef_hindsight_goal_pos_w = np.zeros((3, 3), dtype=np.float32)
        self._compliance = self._sample_compliance()

        self._motion_aligned = False
        self._history_initialized = False
        self._default_pose_mode = self._start_in_default_pose
        self._motion_done = False
        self._debug_dump_count = 0
        self._debug_dump_limit = int(os.getenv("CHIP_DEBUG_DUMP_STEPS", "0"))
        self._debug_dump_path = os.getenv("CHIP_DEBUG_DUMP_PATH", "")

    def post_step_callback(self, commands=None):
        for cmd in commands or []:
            if cmd in ("[MOTION_RESET]", "[MOTION_FADE_IN]"):
                self.reset()
                self._default_pose_mode = False
                return
            if cmd == "[MOTION_FADE_OUT]":
                self._default_pose_mode = True
                self.last_action = np.zeros(self.num_actions, dtype=np.float32)
                return
        if not self._default_pose_mode:
            self._frame += 1
            self._time = self._frame * self.dt
            if self._motion_time_end is not None:
                motion_window = max(self._motion_time_end - self._motion_time_start, 0.0)
                self._motion_done = self._time >= motion_window

    def set_default_pose_mode(self, enabled: bool):
        self._default_pose_mode = bool(enabled)
        if enabled:
            self.last_action = np.zeros(self.num_actions, dtype=np.float32)
            self._motion_done = False

    def reset_alignment(self):
        self._motion_aligned = False
        self._chip_goal_initialized = False
        self._motion_done = False

    def _ensure_motion_aligned(self, env_data):
        if self._motion_aligned:
            return
        root_pos = np.asarray(env_data.base_pos if env_data.base_pos is not None else np.zeros(3), dtype=np.float32)
        self._motion.reset(
            root_pos_w=root_pos,
            t=self._motion_time_start,
            time_start=self._motion_time_start,
            time_end=self._motion_time_end,
        )
        self._motion_aligned = True

    def _get_root_frame(self, env_data) -> tuple[np.ndarray, np.ndarray]:
        """Return the free-joint root frame used by CHIP keypoint observations."""
        root_pos = np.asarray(env_data.base_pos if env_data.base_pos is not None else np.zeros(3), dtype=np.float32)
        root_quat = self._normalize_quat(np.asarray(env_data.base_quat, dtype=np.float32))
        return root_pos, root_quat

    def get_motion_init_qpos_xyzw(self) -> np.ndarray:
        """Return [root_pos, root_quat_xyzw, joint_pos] for motion frame 0."""
        self._motion.update(
            0.0,
            time_start=self._motion_time_start,
            time_end=self._motion_time_end,
        )
        root_pos = self._motion.raw_body_pos(self._body_ids["pelvis"]).astype(np.float32)
        root_quat = self._motion.body_quat(self._body_ids["pelvis"]).astype(np.float32)
        joint_pos = self._motion.joint_pos_at().astype(np.float32)
        return np.concatenate([root_pos, root_quat, joint_pos], axis=0).astype(np.float32)

    def get_motion_init_qvel(self) -> np.ndarray:
        """Return [root_lin_vel_w, root_ang_vel_w, joint_vel] for motion frame 0."""
        self._motion.update(
            0.0,
            time_start=self._motion_time_start,
            time_end=self._motion_time_end,
        )
        root_lin_vel = self._motion.body_lin_vel(self._body_ids["pelvis"]).astype(np.float32)
        root_ang_vel = self._motion.body_ang_vel(self._body_ids["pelvis"]).astype(np.float32)
        joint_vel = self._motion.joint_vel_at().astype(np.float32)
        return np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=0).astype(np.float32)

    def _build_obs(self, env_data) -> np.ndarray:
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        root_pos, root_quat = self._get_root_frame(env_data)

        if getattr(env_data, "imu_lin_vel", None) is not None:
            base_lin_vel_b = np.asarray(env_data.imu_lin_vel, dtype=np.float32)
        elif env_data.base_lin_vel is not None:
            base_lin_vel_b = np.asarray(env_data.base_lin_vel, dtype=np.float32)
        else:
            base_lin_vel_b = np.zeros(3, dtype=np.float32)
        base_ang_vel_b = np.asarray(env_data.base_ang_vel, dtype=np.float32)

        torso_pos_w, torso_quat = self._get_anchor_pose(env_data, root_pos, root_quat)

        self._motion.update(
            self._time,
            time_start=self._motion_time_start,
            time_end=self._motion_time_end,
        )
        motion_pos = self._motion.joint_pos_at()
        motion_vel = self._motion.joint_vel_at()

        command = np.concatenate([motion_pos[self._cmd_joint_ids], motion_vel[self._cmd_joint_ids]], axis=0).astype(
            np.float32
        )

        motion_anchor_pos_w = self._motion.body_pos(self._body_ids["torso_link"])
        motion_anchor_quat = self._motion.body_quat(self._body_ids["torso_link"])
        anchor_pos_b = quat_rotate_inverse_np(torso_quat, motion_anchor_pos_w - torso_pos_w).astype(np.float32)
        anchor_ori_b = _quat_to_mat_first_two_cols(_quat_mul_xyzw(_quat_conj_xyzw(torso_quat), motion_anchor_quat))

        joint_pos_rel = (dof_pos + self._joint_pos_bias) - self._default_dof_pos
        joint_vel_rel = dof_vel.copy()
        if not self._history_initialized:
            self._hist_base_lin_vel = deque(
                [base_lin_vel_b.copy() for _ in range(self.history_length)],
                maxlen=self.history_length,
            )
            self._hist_base_ang_vel = deque(
                [base_ang_vel_b.copy() for _ in range(self.history_length)],
                maxlen=self.history_length,
            )
            self._hist_joint_pos_rel = deque(
                [joint_pos_rel.copy() for _ in range(self.history_length)],
                maxlen=self.history_length,
            )
            self._hist_joint_vel_rel = deque(
                [joint_vel_rel.copy() for _ in range(self.history_length)],
                maxlen=self.history_length,
            )
            self._hist_action = deque(
                [self.last_action.copy() for _ in range(self.history_length)],
                maxlen=self.history_length,
            )
            self._history_initialized = True
        else:
            self._hist_base_lin_vel.append(base_lin_vel_b)
            self._hist_base_ang_vel.append(base_ang_vel_b)
            self._hist_joint_pos_rel.append(joint_pos_rel)
            self._hist_joint_vel_rel.append(joint_vel_rel)
            self._hist_action.append(self.last_action.copy())

        hist_base_lin = np.concatenate(list(self._hist_base_lin_vel), axis=0)
        hist_base_ang = np.concatenate(list(self._hist_base_ang_vel), axis=0)
        hist_jpos = np.concatenate(list(self._hist_joint_pos_rel), axis=0)
        hist_jvel = np.concatenate(list(self._hist_joint_vel_rel), axis=0)
        hist_act = np.concatenate(list(self._hist_action), axis=0)

        ref_pos_terms = []
        ref_ori_terms = []
        for i, body_id in enumerate(self._key_body_ids):
            ref_pos_w = self._motion.body_pos(body_id)
            ref_quat_w = self._motion.body_quat(body_id)
            if not self._chip_goal_initialized:
                self._eef_goal_pos_w[i] = ref_pos_w
                self._eef_hindsight_goal_pos_w[i] = ref_pos_w
            else:
                target = self._goal_alpha * ref_pos_w + (1.0 - self._goal_alpha) * self._eef_goal_pos_w[i]
                self._eef_goal_pos_w[i] = target
                self._eef_hindsight_goal_pos_w[i] = target

            p_b = quat_rotate_inverse_np(root_quat, self._eef_hindsight_goal_pos_w[i] - root_pos).astype(np.float32)
            q_b = _quat_to_mat_first_two_cols(_quat_mul_xyzw(_quat_conj_xyzw(root_quat), ref_quat_w))
            ref_pos_terms.append(p_b)
            ref_ori_terms.append(q_b)

        self._chip_goal_initialized = True

        obs = np.concatenate(
            [
                command,
                anchor_pos_b,
                anchor_ori_b,
                hist_base_lin,
                hist_base_ang,
                hist_jpos,
                hist_jvel,
                hist_act,
                self._compliance,
                np.concatenate(ref_pos_terms, axis=0),
                np.concatenate(ref_ori_terms, axis=0),
            ],
            axis=0,
        ).astype(np.float32)

        if self._debug_dump_limit > 0 and self._debug_dump_count < self._debug_dump_limit:
            self._dump_debug_step(
                command=command,
                anchor_pos_b=anchor_pos_b,
                anchor_ori_b=anchor_ori_b,
                hist_base_lin=hist_base_lin,
                hist_base_ang=hist_base_ang,
                hist_jpos=hist_jpos,
                hist_jvel=hist_jvel,
                hist_act=hist_act,
                compliance=self._compliance,
                ref_pos=np.concatenate(ref_pos_terms, axis=0),
                ref_ori=np.concatenate(ref_ori_terms, axis=0),
                obs=obs,
            )
            self._debug_dump_count += 1
        return obs

    def _dump_debug_step(
        self,
        command: np.ndarray,
        anchor_pos_b: np.ndarray,
        anchor_ori_b: np.ndarray,
        hist_base_lin: np.ndarray,
        hist_base_ang: np.ndarray,
        hist_jpos: np.ndarray,
        hist_jvel: np.ndarray,
        hist_act: np.ndarray,
        compliance: np.ndarray,
        ref_pos: np.ndarray,
        ref_ori: np.ndarray,
        obs: np.ndarray,
    ):
        if not self._debug_dump_path:
            return
        base = Path(self._debug_dump_path)
        base.mkdir(parents=True, exist_ok=True)
        out_path = base / f"chip_obs_step_{self._debug_dump_count:04d}.npz"
        np.savez_compressed(
            out_path,
            frame=np.asarray([self._frame], dtype=np.int32),
            time=np.asarray([self._time], dtype=np.float32),
            command=command.astype(np.float32),
            anchor_pos_b=anchor_pos_b.astype(np.float32),
            anchor_ori_b=anchor_ori_b.astype(np.float32),
            hist_base_lin=hist_base_lin.astype(np.float32),
            hist_base_ang=hist_base_ang.astype(np.float32),
            hist_joint_pos=hist_jpos.astype(np.float32),
            hist_joint_vel=hist_jvel.astype(np.float32),
            hist_action=hist_act.astype(np.float32),
            compliance=compliance.astype(np.float32),
            keypoint_pos=ref_pos.astype(np.float32),
            keypoint_ori=ref_ori.astype(np.float32),
            obs_raw=obs.astype(np.float32),
        )

    def get_observation(self, env_data, ctrl_data):
        self._ensure_motion_aligned(env_data)
        obs = self._build_obs(env_data)
        if obs.shape[0] != self._obs_dim:
            raise RuntimeError(f"CHIP obs dim mismatch: got {obs.shape[0]}, expected {self._obs_dim}")
        obs_n = (obs - self._obs_mean) / self._obs_std
        if not self._default_pose_mode and self._motion_time_end is not None:
            motion_window = max(self._motion_time_end - self._motion_time_start, 0.0)
            self._motion_done = self._time >= motion_window
        callbacks = ["[MOTION_DONE]"] if self._motion_done and not self._default_pose_mode else []
        return obs_n, {"CALLBACK": callbacks}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self._default_pose_mode:
            self.last_action = np.zeros(self.num_actions, dtype=np.float32)
            return np.zeros(self.num_actions, dtype=np.float32)

        obs_t = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_action = self._actor(obs_t).cpu().numpy().squeeze(0).astype(np.float32)
        raw_action = np.clip(raw_action, -100.0, 100.0)
        action_delta = raw_action * self._joint_action_scale
        self.last_action = raw_action.copy()
        return action_delta

    def get_init_dof_pos(self) -> np.ndarray:
        return self._default_dof_pos.copy()

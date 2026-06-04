"""Unit tests for CHIP impulse event robustness."""

from types import SimpleNamespace

import torch

from mjlab.tasks.tracking.mdp.chip import apply_chip_eef_impulse


class _DummyAsset:
  def __init__(self, num_envs: int, num_bodies: int, device: str):
    self.num_bodies = num_bodies
    self.data = SimpleNamespace(
      body_external_wrench=torch.zeros(num_envs, num_bodies, 6, device=device),
      body_com_pos_w=torch.zeros(num_envs, num_bodies, 3, device=device),
    )
    self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | slice | None, object]] = []

  def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None):
    self.calls.append((forces.clone(), torques.clone(), env_ids, body_ids))


class _DummyEnv:
  def __init__(self, num_envs: int, num_bodies: int, device: str = "cpu"):
    self.num_envs = num_envs
    self.device = device
    self.step_dt = 0.02
    self.scene = {"robot": _DummyAsset(num_envs, num_bodies, device)}


def _make_cfg(body_ids):
  return SimpleNamespace(
    params={
      "asset_cfg": SimpleNamespace(name="robot", body_ids=body_ids),
      "cooldown_s": (0.0, 0.0),
    }
  )


def test_num_bodies_matches_tensor_body_ids() -> None:
  env = _DummyEnv(num_envs=4, num_bodies=20)
  event = apply_chip_eef_impulse(_make_cfg(torch.tensor([4, 5])), env)

  assert event._num_bodies == 2
  assert event._peak_forces.shape == (4, 2, 3)


def test_step_respects_env_ids_mask() -> None:
  env = _DummyEnv(num_envs=4, num_bodies=20)
  event = apply_chip_eef_impulse(_make_cfg(torch.tensor([4, 5])), env)

  # Prepare one active env outside step mask and one inside step mask.
  event._active[:] = False
  event._active[0] = True
  event._active[1] = True
  event._time_remaining[:] = 1.0
  event._duration[:] = 1.0
  prev_time = event._time_remaining.clone()

  # Only envs [1, 2] should be updated.
  event(
    env=env,
    env_ids=torch.tensor([1, 2]),
    force_range=(0.0, 0.0),
    duration_s=(0.1, 0.1),
    cooldown_s=(0.1, 0.1),
    asset_cfg=SimpleNamespace(name="robot", body_ids=torch.tensor([4, 5])),
  )

  # Env 1 time decreased; env 0 untouched although active.
  assert event._time_remaining[1] < prev_time[1]
  assert torch.allclose(event._time_remaining[0], prev_time[0])


def test_reset_slice_partial_does_not_overflow() -> None:
  env = _DummyEnv(num_envs=16, num_bodies=20)
  event = apply_chip_eef_impulse(_make_cfg(torch.tensor([4, 5])), env)

  # Should not throw for partial slice (was a crash case).
  event.reset(slice(0, 5))

  assert event._cooldown_remaining[:5].shape == (5,)

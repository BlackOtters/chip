"""Unit tests for CHIP command target wiring."""

import torch

from mjlab.tasks.tracking.mdp.chip import ChipMotionCommand


def test_hindsight_targets_replace_only_selected_eef_positions() -> None:
  command = object.__new__(ChipMotionCommand)
  command.body_pos_relative_w = torch.tensor(
    [
      [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
      ],
      [
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
      ],
    ],
    dtype=torch.float32,
  )
  command.eef_motion_indexes = torch.tensor([0, 2], dtype=torch.long)
  command.eef_hindsight_goal_pos_w = torch.tensor(
    [
      [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
      ],
      [
        [30.0, 30.0, 30.0],
        [40.0, 40.0, 40.0],
      ],
    ],
    dtype=torch.float32,
  )

  ChipMotionCommand._apply_hindsight_body_targets(command)

  expected = torch.tensor(
    [
      [
        [10.0, 10.0, 10.0],
        [1.0, 1.0, 1.0],
        [20.0, 20.0, 20.0],
      ],
      [
        [30.0, 30.0, 30.0],
        [4.0, 4.0, 4.0],
        [40.0, 40.0, 40.0],
      ],
    ],
    dtype=torch.float32,
  )
  assert torch.allclose(command.body_pos_relative_w, expected)

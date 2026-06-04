"""RL configuration for Unitree G1 tracking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

CHIP_ATTENTION_MODEL = "mjlab.rl.chip_attention_model:ChipTemporalCrossAttentionModel"


def _chip_attention_cfg() -> dict:
  return {
    "history_length": 10,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 1,
    "ff_dim": 256,
    "dropout": 0.0,
  }


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_chip_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 CHIP tracking task."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_chip_tracking_attn"
  cfg.obs_groups = {
    "actor": ("actor",),
    "critic": ("critic",),
  }
  cfg.actor.class_name = CHIP_ATTENTION_MODEL
  cfg.actor.attention_cfg = _chip_attention_cfg()
  return cfg

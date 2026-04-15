import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
POLICY_NAME = "default"


def _resolve_checkpoint_path() -> str:
    """
    Resolve checkpoint location in this order:
    1) env var STRONG_AGENT_CHECKPOINT
    2) packaged checkpoint path under this module
    """
    env_path = os.environ.get("STRONG_AGENT_CHECKPOINT")
    if env_path and os.path.exists(env_path):
        return env_path

    module_dir = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(module_dir, "checkpoint", "checkpoint")
    if os.path.exists(packaged):
        return packaged

    raise FileNotFoundError(
        "Strong agent checkpoint not found. Set STRONG_AGENT_CHECKPOINT or package"
        " checkpoint to my_strong_agent/checkpoint/checkpoint"
    )


def _load_rllib_config(checkpoint_path: str) -> dict:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    direct = os.path.join(checkpoint_dir, "params.pkl")
    parent = os.path.join(checkpoint_dir, "..", "params.pkl")

    config_path = direct if os.path.exists(direct) else parent
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find params.pkl near checkpoint: {checkpoint_path}"
        )

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    # Inference-only settings.
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["num_envs_per_worker"] = 1
    config["log_level"] = "ERROR"

    # No real env is needed to run policy inference from checkpoint.
    tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
    config["env"] = "DummyEnv"

    return config


class StrongPPOAgent(AgentInterface):
    """
    Agent wrapper used for soccer_twos evaluation/submission.
    Loads PPO checkpoint and computes actions for each controlled player.
    """

    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = _resolve_checkpoint_path()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        config = _load_rllib_config(checkpoint_path)
        trainer_cls = get_trainable_cls(ALGORITHM)
        trainer = trainer_cls(env=config["env"], config=config)
        trainer.restore(checkpoint_path)

        self.policy = trainer.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            action, *_ = self.policy.compute_single_action(obs, explore=False)
            actions[player_id] = action
        return actions

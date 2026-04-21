import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from utils import create_rllib_env


ALGORITHM = "PPO"
POLICY_NAME = "default"
DEFAULT_POLICY_NAME = "default_policy"


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
        " checkpoint to my_curriculum_agent/checkpoint/checkpoint"
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

    # Build a temp env once to recover observation/action spaces for single-agent PPO.
    env_cfg = dict(config.get("env_config", {}))
    env_cfg["base_port"] = int(env_cfg.get("base_port", 15000)) + (os.getpid() % 10000)
    temp_env = create_rllib_env(env_cfg)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    class SpaceOnlyEnv(gym.Env):
        def __init__(self, observation_space, action_space):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space

        def reset(self):
            return self.observation_space.sample()

        def step(self, action):
            del action
            return self.observation_space.sample(), 0.0, True, {}

    tune.registry.register_env(
        "DummyEnv", lambda *_: SpaceOnlyEnv(obs_space, act_space)
    )
    config["env"] = "DummyEnv"

    return config


class StrongPPOAgent(AgentInterface):
    """
    Agent wrapper used for soccer_twos evaluation/submission.
    Loads PPO checkpoint and computes actions for each controlled player.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        del env

        checkpoint_path = _resolve_checkpoint_path()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        config = _load_rllib_config(checkpoint_path)
        trainer_cls = get_trainable_cls(ALGORITHM)
        trainer = trainer_cls(env=config["env"], config=config)
        trainer.restore(checkpoint_path)

        self.policy = trainer.get_policy(POLICY_NAME)
        if self.policy is None:
            self.policy = trainer.get_policy(DEFAULT_POLICY_NAME)
        if self.policy is None:
            # Some configs expose only one unnamed/default policy.
            self.policy = trainer.get_policy()
        if self.policy is None:
            raise RuntimeError("Could not load PPO policy from checkpoint")

        obs_space = getattr(self.policy, "observation_space", None)
        self._expected_obs_dim = None
        if obs_space is not None and hasattr(obs_space, "shape") and obs_space.shape:
            self._expected_obs_dim = int(obs_space.shape[0])

        act_space = getattr(self.policy, "action_space", None)
        self._team_action_mode = False
        if isinstance(act_space, gym.spaces.MultiDiscrete):
            nvec = np.asarray(act_space.nvec).reshape(-1)
            # team_vs_policy with two players usually emits 6 discrete branches.
            self._team_action_mode = int(nvec.shape[0]) >= 6

    def _align_obs_dim(self, obs: np.ndarray) -> np.ndarray:
        if self._expected_obs_dim is None:
            return np.asarray(obs, dtype=np.float32)

        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        cur_dim = int(obs_arr.shape[0])

        if cur_dim == self._expected_obs_dim:
            return obs_arr

        if cur_dim < self._expected_obs_dim:
            pad = np.zeros(self._expected_obs_dim - cur_dim, dtype=np.float32)
            return np.concatenate([obs_arr, pad], axis=0)

        return obs_arr[: self._expected_obs_dim]

    @staticmethod
    def _as_discrete_action_vec(action) -> np.ndarray:
        if isinstance(action, tuple):
            action = action[0]
        return np.asarray(action, dtype=np.int64).reshape(-1)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # Curriculum checkpoint was trained with concatenated team observations.
        if self._team_action_mode and 0 in observation and 1 in observation:
            team_obs = np.concatenate(
                [
                    np.asarray(observation[0], dtype=np.float32).reshape(-1),
                    np.asarray(observation[1], dtype=np.float32).reshape(-1),
                ],
                axis=0,
            )
            aligned_team_obs = self._align_obs_dim(team_obs)
            team_action, *_ = self.policy.compute_single_action(
                aligned_team_obs, explore=False
            )
            team_action = self._as_discrete_action_vec(team_action)

            if team_action.shape[0] >= 6:
                return {
                    0: team_action[:3],
                    1: team_action[3:6],
                }

        actions = {}
        for player_id, obs in observation.items():
            aligned_obs = self._align_obs_dim(obs)
            action, *_ = self.policy.compute_single_action(aligned_obs, explore=False)
            actions[player_id] = action
        return actions

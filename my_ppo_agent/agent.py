from typing import Dict
import os
import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface, EnvType
import soccer_twos
from soccer_twos import AgentInterface, EnvType



class DummyEnv(gym.Env):
    def __init__(self, config=None):
        config = config or {}
        self.observation_space = config["observation_space"]
        self.action_space = config["action_space"]

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


class PPOAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        # keep observation space from watch env
        self.observation_space = env.observation_space

        # IMPORTANT:
        # Training used flatten_branched=True, which flattened actions.
        # Your checkpoint expects a Discrete(27) output layer.
        self.action_space = gym.spaces.Discrete(27)

        self.checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "checkpoint_000835",
            "checkpoint-835",
        )

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}"
            )

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                num_cpus=1,
                local_mode=True,
            )

        try:
            tune.registry.register_env(
                "dummy_soccer_env",
                lambda cfg: DummyEnv(cfg)
            )
        except Exception:
            pass

        self.trainer = PPOTrainer(
            env="dummy_soccer_env",
            config={
                "framework": "torch",
                "num_gpus": 0,
                "num_workers": 0,
                "num_envs_per_worker": 1,
                "log_level": "ERROR",
                "env_config": {
                    "observation_space": self.observation_space,
                    "action_space": self.action_space,
                },
                "model": {
                    "vf_share_layers": True,
                    "fcnet_hiddens": [512],
                },
            },
        )

        self.trainer.restore(self.checkpoint_path)

    def _unflatten_action(self, action: int) -> np.ndarray:
        # Convert Discrete(27) back to 3 branched actions of size 3 each
        return np.array(np.unravel_index(int(action), (3, 3, 3)), dtype=np.int32)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            flat_action = self.trainer.compute_action(obs)
            actions[player_id] = self._unflatten_action(flat_action)
        return actions



if __name__ == "__main__":
    env = soccer_twos.make(
        watch=False,
        flatten_branched=True,
        variation=EnvType.team_vs_policy,
        single_player=True,
        opponent_policy=lambda *_: 0,
    )

    print("Real Observation Space:", env.observation_space)
    print("Real Action Space:", env.action_space)

    agent = PPOAgent(env)

    fake_obs = {
        0: np.zeros(env.observation_space.shape, dtype=np.float32),
        1: np.zeros(env.observation_space.shape, dtype=np.float32),
    }

    actions = agent.act(fake_obs)
    print("Actions:", actions)
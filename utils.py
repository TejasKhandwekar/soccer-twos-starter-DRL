from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class BallProgressRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    Adds dense shaping reward based on whether the ball moves toward each team's
    scoring direction along the x-axis.

    Team 0 attacks +x, Team 1 attacks -x.
    """

    def __init__(self, env, progress_weight=0.03, clip_abs=0.05):
        super().__init__(env)
        self.progress_weight = float(progress_weight)
        self.clip_abs = float(clip_abs)
        self._last_ball_x = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_ball_x = None
        return obs

    @staticmethod
    def _extract_ball_x(info):
        if not isinstance(info, dict) or not info:
            return None
        for _, agent_info in info.items():
            if not isinstance(agent_info, dict):
                continue
            ball_info = agent_info.get("ball_info", {})
            if not isinstance(ball_info, dict):
                continue
            position = ball_info.get("position")
            if isinstance(position, (list, tuple)) and len(position) >= 1:
                return float(position[0])
        return None

    @staticmethod
    def _team_sign(agent_id):
        # agent ids 0,1 are team 0 (+x attack); 2,3 are team 1 (-x attack)
        return 1.0 if agent_id in (0, 1) else -1.0

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        if not isinstance(rewards, dict):
            return obs, rewards, done, info

        ball_x = self._extract_ball_x(info)
        if ball_x is None:
            return obs, rewards, done, info

        if self._last_ball_x is None:
            self._last_ball_x = ball_x
            return obs, rewards, done, info

        delta_x = ball_x - self._last_ball_x
        self._last_ball_x = ball_x

        shaped = dict(rewards)
        for agent_id, base_reward in rewards.items():
            directed_progress = self._team_sign(agent_id) * delta_x
            bonus = self.progress_weight * directed_progress
            bonus = max(-self.clip_abs, min(self.clip_abs, bonus))
            shaped[agent_id] = float(base_reward) + bonus

        return obs, shaped, done, info


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
            - use_ball_progress_reward: bool, enables dense reward shaping wrapper.
            - ball_progress_reward_config: dict for shaping config.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)

    if env_config.get("use_ball_progress_reward", False):
        reward_cfg = env_config.get("ball_progress_reward_config", {})
        env = BallProgressRewardWrapper(env, **reward_cfg)

    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s

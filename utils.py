from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class BallProgressRewardWrapper(gym.core.Wrapper):
    """
    Comprehensive dense reward shaping:
    1. Ball progress toward opponent goal (offensive)
    2. Ball territory (ball in enemy half)
    3. Possession incentive (ball proximity)
    4. Defense incentive (staying near own goal when under pressure)
    5. Conceding penalty (negative when opponent scores)

    Team 0 attacks +x, Team 1 attacks -x.
    """

    def __init__(self, env, progress_weight=0.08, territory_weight=0.01, 
                 possession_weight=0.02, defense_weight=0.01, concede_penalty=1.0, clip_abs=0.20):
        super().__init__(env)
        self.progress_weight = float(progress_weight)
        self.territory_weight = float(territory_weight)
        self.possession_weight = float(possession_weight)
        self.defense_weight = float(defense_weight)
        self.concede_penalty = float(concede_penalty)
        self.clip_abs = float(clip_abs)
        self._last_ball_x = None
        self._last_scores = {0: 0, 1: 0}  # Track goals for both teams

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_ball_x = None
        self._last_scores = {0: 0, 1: 0}
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

    @staticmethod
    def _get_team_id(agent_id):
        return 0 if agent_id in (0, 1) else 1

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        # team_vs_policy returns a scalar reward instead of per-agent dict.
        if not isinstance(rewards, dict):
            ball_x = self._extract_ball_x(info)
            if ball_x is None:
                return obs, rewards, done, info

            if self._last_ball_x is None:
                self._last_ball_x = ball_x
                return obs, rewards, done, info

            delta_x = ball_x - self._last_ball_x
            self._last_ball_x = ball_x

            # In team_vs_policy, learning controls team 0 (attacks +x).
            directed_progress = delta_x
            directed_ball_x = ball_x

            # Try to infer scoreboard in team_vs_policy info payload.
            current_scores = dict(self._last_scores)
            if isinstance(info, dict):
                for _, maybe_info in info.items():
                    if not isinstance(maybe_info, dict):
                        continue
                    team_id = maybe_info.get("team_id")
                    score = maybe_info.get("score")
                    if team_id in (0, 1) and isinstance(score, (int, float)):
                        current_scores[int(team_id)] = int(score)

            concede_bonus = 0.0
            # Learning side is team 0 in team_vs_policy mode.
            if current_scores.get(1, 0) > self._last_scores.get(1, 0):
                concede_bonus = -self.concede_penalty

            bonus = (
                self.progress_weight * directed_progress
                + self.territory_weight * max(0.0, directed_ball_x)
                + self.possession_weight * abs(delta_x)
                - self.defense_weight * max(0.0, -directed_ball_x) * 0.1
                + concede_bonus
            )
            bonus = max(-self.clip_abs, min(self.clip_abs, bonus))
            self._last_scores = current_scores
            return obs, float(rewards) + bonus, done, info

        ball_x = self._extract_ball_x(info)
        if ball_x is None:
            return obs, rewards, done, info

        if self._last_ball_x is None:
            self._last_ball_x = ball_x
            return obs, rewards, done, info

        delta_x = ball_x - self._last_ball_x
        self._last_ball_x = ball_x

        # Extract current scores if available
        current_scores = {0: 0, 1: 0}
        for agent_info in info.values():
            if isinstance(agent_info, dict):
                team_id = agent_info.get("team_id")
                if team_id in (0, 1):
                    current_scores[team_id] = agent_info.get("score", 0)

        shaped = dict(rewards)
        for agent_id, base_reward in rewards.items():
            team_sign = self._team_sign(agent_id)
            team_id = self._get_team_id(agent_id)
            opponent_id = 1 - team_id

            # 1. Ball progress: reward moving ball toward opponent goal
            directed_progress = team_sign * delta_x
            progress_bonus = self.progress_weight * directed_progress

            # 2. Territory: reward ball in offensive half (positive x for team0, negative x for team1)
            directed_ball_x = team_sign * ball_x
            territory_bonus = self.territory_weight * max(0, directed_ball_x)  # Only reward advancing half

            # 3. Possession incentive: encourage ball engagement
            # Rough proxy: ball moving suggests active play
            possession_bonus = self.possession_weight * abs(delta_x)

            # 4. Defense incentive: penalize being far from own goal when opponent has ball
            # (This can be refined further with positional info if available)
            defense_bonus = -self.defense_weight * max(0, -directed_ball_x) * 0.1  # Light penalty when ball far from goal

            # 5. Conceding penalty: strong negative reward when opponent scores
            concede_bonus = 0.0
            if current_scores[opponent_id] > self._last_scores[opponent_id]:
                concede_bonus = -self.concede_penalty

            bonus = progress_bonus + territory_bonus + possession_bonus + defense_bonus + concede_bonus
            bonus = max(-self.clip_abs, min(self.clip_abs, bonus))
            shaped[agent_id] = float(base_reward) + bonus

        self._last_scores = current_scores
        return obs, shaped, done, info


class BallFeatureObservationWrapper(gym.core.Wrapper):
    """
    Appends two dense features to each observation vector:
    1) Team-directed ball x position
    2) Team-directed ball x delta (step-to-step)

    This enriches state with compact game-phase information that is aligned with
    each team's scoring direction.
    """

    def __init__(self, env, feature_clip=1.0):
        super().__init__(env)
        self.feature_clip = float(feature_clip)
        self._last_ball_x = None

        base_space = getattr(env, "observation_space", None)
        self._augment_enabled = isinstance(base_space, gym.spaces.Box) and len(base_space.shape) == 1

        if self._augment_enabled:
            ext_low = np.concatenate(
                [
                    np.asarray(base_space.low, dtype=np.float32),
                    np.array([-self.feature_clip, -self.feature_clip], dtype=np.float32),
                ]
            )
            ext_high = np.concatenate(
                [
                    np.asarray(base_space.high, dtype=np.float32),
                    np.array([self.feature_clip, self.feature_clip], dtype=np.float32),
                ]
            )
            self.observation_space = gym.spaces.Box(low=ext_low, high=ext_high, dtype=np.float32)

    @staticmethod
    def _team_sign(agent_id):
        return 1.0 if agent_id in (0, 1) else -1.0

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

    def _augment_vec(self, obs_vec, team_sign, ball_x, ball_dx):
        if not self._augment_enabled:
            return obs_vec
        d_ball_x = float(np.clip(team_sign * ball_x, -self.feature_clip, self.feature_clip))
        d_ball_dx = float(np.clip(team_sign * ball_dx, -self.feature_clip, self.feature_clip))
        return np.concatenate(
            [
                np.asarray(obs_vec, dtype=np.float32),
                np.array([d_ball_x, d_ball_dx], dtype=np.float32),
            ]
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_ball_x = None

        if not self._augment_enabled:
            return obs

        if isinstance(obs, dict):
            return {
                aid: self._augment_vec(avec, self._team_sign(aid), 0.0, 0.0)
                for aid, avec in obs.items()
            }

        return self._augment_vec(obs, 1.0, 0.0, 0.0)

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        if not self._augment_enabled:
            return obs, rewards, done, info

        ball_x = self._extract_ball_x(info)
        if ball_x is None:
            ball_x = self._last_ball_x if self._last_ball_x is not None else 0.0

        if self._last_ball_x is None:
            ball_dx = 0.0
        else:
            ball_dx = ball_x - self._last_ball_x
        self._last_ball_x = ball_x

        if isinstance(obs, dict):
            aug_obs = {
                aid: self._augment_vec(avec, self._team_sign(aid), ball_x, ball_dx)
                for aid, avec in obs.items()
            }
        else:
            aug_obs = self._augment_vec(obs, 1.0, ball_x, ball_dx)

        return aug_obs, rewards, done, info


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
            - use_ball_feature_observation: bool, appends directed ball features to each observation.
            - ball_feature_observation_config: dict for observation feature config.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    make_kwargs = dict(env_config)
    base_port = make_kwargs.get("base_port")
    max_port_retries = int(make_kwargs.pop("max_port_retries", 5))

    env = None
    last_err = None
    for attempt in range(max_port_retries):
        try:
            env = soccer_twos.make(**make_kwargs)
            break
        except Exception as e:
            msg = str(e)
            is_port_conflict = (
                "UnityWorkerInUseException" in msg
                or "Address already in use" in msg
                or "worker number" in msg
            )
            if not is_port_conflict or base_port is None or attempt == max_port_retries - 1:
                raise

            # Shift to a new port block to avoid collisions from other jobs.
            shifted_base_port = int(base_port) + (attempt + 1) * 1000
            make_kwargs["base_port"] = shifted_base_port
            last_err = e

    if env is None and last_err is not None:
        raise last_err

    if env_config.get("use_ball_progress_reward", False):
        reward_cfg = env_config.get("ball_progress_reward_config", {})
        env = BallProgressRewardWrapper(env, **reward_cfg)

    if env_config.get("use_ball_feature_observation", False):
        obs_cfg = env_config.get("ball_feature_observation_config", {})
        env = BallFeatureObservationWrapper(env, **obs_cfg)

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

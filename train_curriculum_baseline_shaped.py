import logging
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.tune.logger import NoopLogger
from ray.tune.registry import get_trainable_cls
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 2
TRAINING_HOURS = int(os.environ.get("STRONG_TRAIN_HOURS", "16"))
TIMESTEP_TARGET = int(os.environ.get("STRONG_TRAIN_TIMESTEPS", "30000000"))
RESTORE_CHECKPOINT = os.environ.get("STRONG_RESTORE_CHECKPOINT")
DEFAULT_BASE_PORT = 15000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 40000)
BASE_PORT = int(os.environ.get("STRONG_BASE_PORT", str(DEFAULT_BASE_PORT)))

BASELINE_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent",
    "ray_results",
    "PPO_selfplay_twos",
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02",
    "checkpoint_002449",
    "checkpoint-2449",
)

STRONG_OPPONENT_CKPT = os.environ.get(
    "STRONG_OPPONENT_CKPT",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "my_strong_agent",
        "checkpoint",
        "checkpoint",
    ),
)


class HideAgentCrashFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "The agent on node" in msg or "socket.gaierror" in msg:
            return False
        return True


logging.getLogger("ray._private.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray").setLevel(logging.ERROR)

os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_DISABLE_REPORTER"] = "1"


class CheckpointOpponentPolicy:
    """Lazy-loaded opponent policy callable from an RLlib checkpoint."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self._trainer = None
        self._policy = None
        self._obs_dim = None

    def _load(self):
        if self._policy is not None:
            return

        ckpt_dir = os.path.dirname(self.checkpoint_path)
        params_path = os.path.join(ckpt_dir, "params.pkl")
        if not os.path.exists(params_path):
            params_path = os.path.join(ckpt_dir, "..", "params.pkl")

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"params.pkl not found near {self.checkpoint_path}")

        with open(params_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["num_envs_per_worker"] = 1
        config["log_level"] = "ERROR"

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        trainer_cls = get_trainable_cls("PPO")
        self._trainer = trainer_cls(env=config["env"], config=config)
        self._trainer.restore(self.checkpoint_path)
        self._policy = self._trainer.get_policy("default")
        if self._policy is None:
            self._policy = self._trainer.get_policy("default_policy")
        if self._policy is None:
            self._policy = self._trainer.get_policy()

        if self._policy is None:
            raise RuntimeError(f"Could not load policy from checkpoint: {self.checkpoint_path}")

        obs_space = getattr(self._policy, "observation_space", None)
        if obs_space is not None and getattr(obs_space, "shape", None):
            self._obs_dim = int(obs_space.shape[0])

    def __call__(self, obs):
        self._load()

        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._obs_dim is not None:
            if obs_arr.shape[0] < self._obs_dim:
                obs_arr = np.concatenate(
                    [obs_arr, np.zeros(self._obs_dim - obs_arr.shape[0], dtype=np.float32)],
                    axis=0,
                )
            elif obs_arr.shape[0] > self._obs_dim:
                obs_arr = obs_arr[: self._obs_dim]

        action, *_ = self._policy.compute_single_action(obs_arr, explore=False)
        return action


def random_opponent_policy(obs):
    del obs
    # TeamVsPolicyWrapper action is MultiDiscrete([3,3,3]) per player.
    return np.array([np.random.randint(0, 3), np.random.randint(0, 3), np.random.randint(0, 3)])


class HybridOpponentPolicy:
    def __init__(self, primary_callable, secondary_callable, primary_prob=0.7):
        self.primary_callable = primary_callable
        self.secondary_callable = secondary_callable
        self.primary_prob = float(primary_prob)

    def __call__(self, obs):
        if np.random.rand() < self.primary_prob:
            return self.primary_callable(obs)
        return self.secondary_callable(obs)


def _set_opponent_policy_in_env(env, policy_fn):
    cursor = env
    for _ in range(12):
        if hasattr(cursor, "set_opponent_policy"):
            cursor.set_opponent_policy(policy_fn)
            return True
        cursor = getattr(cursor, "env", None)
        if cursor is None:
            break
    return False


class CurriculumOpponentCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._stage = None
        self._timesteps_total = 0
        self._strong = CheckpointOpponentPolicy(STRONG_OPPONENT_CKPT)
        self._baseline = CheckpointOpponentPolicy(BASELINE_CKPT)
        self._hybrid = HybridOpponentPolicy(self._baseline, self._strong, primary_prob=0.7)

    def _select_stage(self, timesteps_total):
        # Stage 0: very short warm-up with random opponent.
        if timesteps_total < 200_000:
            return "random", random_opponent_policy
        # Stage 1: short strong-opponent bootstrap.
        if timesteps_total < 4_000_000:
            return "strong", self._strong
        # Stage 2: spend most learning directly against baseline (target metric).
        if timesteps_total < 24_000_000:
            return "baseline", self._baseline
        # Stage 3: robustness tail mixing baseline and strong opponents.
        return "hybrid", self._hybrid

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        del policies, episode, env_index, kwargs

        # Some Ray versions provide worker.global_vars=None at episode start.
        # Use our tracked timesteps (updated in on_train_result) as stable source.
        timesteps_total = int(self._timesteps_total)
        stage_name, policy_fn = self._select_stage(int(timesteps_total))

        for env in base_env.get_unwrapped():
            _set_opponent_policy_in_env(env, policy_fn)

        if stage_name != self._stage:
            self._stage = stage_name
            print(f"[Curriculum] switched stage -> {stage_name} @ timesteps={timesteps_total}")

    def on_train_result(self, *, trainer, result, **kwargs):
        del trainer, kwargs
        self._timesteps_total = int(result.get("timesteps_total", self._timesteps_total))


if __name__ == "__main__":
    os.system("ray stop --force")
    print(f"[Config] STRONG_BASE_PORT={BASE_PORT}")

    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=16,
        num_gpus=1,
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    stop_config = {
        "timesteps_total": TIMESTEP_TARGET,
    }
    if not RESTORE_CHECKPOINT:
        stop_config["time_total_s"] = TRAINING_HOURS * 3600

    tune.run(
        "PPO",
        name="PPO_CURRICULUM_BASELINE_SHAPED",
        loggers=[NoopLogger],
        restore=RESTORE_CHECKPOINT,
        config={
            "num_gpus": 1,
            "num_workers": 10,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": CurriculumOpponentCallback,
            "seed": 42,
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": False,
                "flatten_branched": False,
                "opponent_policy": random_opponent_policy,
                "use_ball_progress_reward": True,
                "ball_progress_reward_config": {
                    "progress_weight": 0.08,
                    "territory_weight": 0.01,
                    "possession_weight": 0.02,
                    "defense_weight": 0.01,
                    "concede_penalty": 1.0,
                    "clip_abs": 0.20,
                },
                "use_ball_feature_observation": False,
            },
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [768, 512, 256],
                "fcnet_activation": "swish",
            },
            "lambda": 0.95,
            "gamma": 0.99,
            "clip_param": 0.2,
            "entropy_coeff": 0.0005,
            "vf_loss_coeff": 1.0,
            # Faster update cadence for quicker learning progress.
            "rollout_fragment_length": 200,
            "train_batch_size": 8000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 12,
            "lr": 5e-4,
            "lr_schedule": [[0, 5e-4], [8_000_000, 1.5e-4], [20_000_000, 5e-5]],
            "batch_mode": "complete_episodes",
        },
        stop=stop_config,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("~/scratch/ray_results"),
    )

    print("Done training")

import logging
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import NoopLogger

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 2
TRAINING_HOURS = int(os.environ.get("STRONG_TRAIN_HOURS", "24"))
TIMESTEP_TARGET = int(os.environ.get("STRONG_TRAIN_TIMESTEPS", "60000000"))
RESTORE_CHECKPOINT = os.environ.get("STRONG_RESTORE_CHECKPOINT")
DEFAULT_BASE_PORT = 15000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 40000)
BASE_PORT = int(os.environ.get("STRONG_BASE_PORT", str(DEFAULT_BASE_PORT)))


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


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id in (0, 1):
        return "default"

    # Team 2 plays from a weighted opponent pool.
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.20, 0.40, 0.25, 0.15],
    )


class SelfPlayArchiveCallback(DefaultCallbacks):
    """
    Periodically promotes the latest trainable policy into opponent archive.
    The threshold is intentionally conservative to avoid destabilizing training.
    """

    def on_train_result(self, *, trainer, result, **kwargs):
        reward_mean = result.get("episode_reward_mean", -999)
        iteration = result.get("training_iteration", 0)

        # Initialize archive opponents from the current trainable policy once,
        # so team-2 does not sample uninitialized/random policy weights.
        if not getattr(self, "_archive_seeded", False):
            default_weights = trainer.get_weights(["default"])["default"]
            trainer.set_weights(
                {
                    "opponent_1": default_weights,
                    "opponent_2": default_weights,
                    "opponent_3": default_weights,
                }
            )
            self._archive_seeded = True
            print("[SelfPlayArchive] seeded opponent archive from default policy")

        # Rotate archive more frequently once the policy reaches stable, non-collapsed play.
        if iteration > 0 and iteration % 25 == 0 and reward_mean > -0.20:
            print(
                f"[SelfPlayArchive] iter={iteration}, reward={reward_mean:.3f} -> rotating opponent weights"
            )
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )


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
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    stop_config = {
        "timesteps_total": TIMESTEP_TARGET,
    }
    if not RESTORE_CHECKPOINT:
        stop_config["time_total_s"] = TRAINING_HOURS * 3600

    analysis = tune.run(
        "PPO",
        name="PPO_STRONG_SELFPLAY_SHAPED",
        loggers=[NoopLogger],
        restore=RESTORE_CHECKPOINT,
        config={
            "num_gpus": 1,
            "num_workers": 14,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": SelfPlayArchiveCallback,
            "seed": 42,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
                # Dense shaping: positive when ball advances toward goal.
                "use_ball_progress_reward": True,
                "ball_progress_reward_config": {
                    "progress_weight": 0.05,
                    "territory_weight": 0.003,
                    "clip_abs": 0.07,
                },
                # Observation augmentation with directed ball-state features.
                "use_ball_feature_observation": True,
                "ball_feature_observation_config": {
                    "feature_clip": 1.0,
                },
            },
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [768, 512, 256],
                "fcnet_activation": "swish",
            },
            "lambda": 0.95,
            "gamma": 0.99,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "vf_loss_coeff": 1.0,
            "rollout_fragment_length": 500,
            "train_batch_size": 16000,
            "sgd_minibatch_size": 1024,
            "num_sgd_iter": 24,
            "batch_mode": "complete_episodes",
            "lr": 3e-4,
            "lr_schedule": [
                [0, 3e-4],
                [15000000, 8e-5],
                [30000000, 3e-5],
            ],
        },
        stop=stop_config,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("~/scratch/ray_results"),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial:
        print(f"Best trial: {best_trial}")
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean",
            mode="max",
        )
        print(f"Best checkpoint: {best_checkpoint}")
    else:
        print("No best trial found (training may have been interrupted).")

    print("Strong self-play training complete.")

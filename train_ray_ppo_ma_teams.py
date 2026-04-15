import os
import logging

# --- SURGICAL LOG SILENCER ---
class HideAgentCrashFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "The agent on node" in msg or "socket.gaierror" in msg:
            return False
        return True


# Apply the silencer to Ray's internal loggers
logging.getLogger("ray._private.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray").setLevel(logging.ERROR)

# Kill Ray background services that caused trouble on PACE
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_DISABLE_REPORTER"] = "1"

import ray
from ray import tune
from ray.tune.logger import NoopLogger
from soccer_twos import EnvType

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 2


if __name__ == "__main__":
    # Cleanup any old Ray processes
    os.system("ray stop --force")

    logging.getLogger("ray").setLevel(logging.ERROR)

    ray.init(
        include_dashboard=False,
        num_cpus=16,
        num_gpus=0,
        log_to_driver=False,
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    # Multi-agent team env: one shared policy per team setup
    temp_env = create_rllib_env({"variation": EnvType.multiagent_team})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_MA_TEAMS",
        loggers=[NoopLogger],
        config={
            # System settings
            "num_gpus": 0,
            "num_workers": 14,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",

            # Multi-agent setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda agent_id: "default"),
                "policies_to_train": ["default"],
            },

            # Environment setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_team,
            },

            # Model architecture
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },

            # PPO batching
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 20000000,
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("~/scratch/ray_results"),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")

    if best_trial:
        print(f"Best Trial found: {best_trial}")
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean",
            mode="max",
        )
        print(f"Best Checkpoint: {best_checkpoint}")
    else:
        print("Training interrupted before a best trial could be determined.")

    print("Process Complete.")

import os
import logging

# --- SURGICAL LOG SILENCER ---
class HideAgentCrashFilter(logging.Filter):
    def filter(self, record):
        # Hide any log message that contains this specific firewall complaint
        if "The agent on node" in record.getMessage() or "socket.gaierror" in record.getMessage():
            return False
        return True

# Apply the silencer to Ray's internal loggers
logging.getLogger("ray._private.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray").setLevel(logging.ERROR)

# 1. Kill Ray's background network and memory services immediately
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
    # 2. Cleanup any "ghost" Ray processes from previous failed runs
    os.system("ray stop --force")

    # 3. Suppress internal Ray networking logs in the console
    logging.getLogger("ray").setLevel(logging.ERROR)

    # 4. Initialize Ray for a single PACE node
    # We use 4 CPUs to match your 'salloc -c 4' request
    ray.init(
        include_dashboard=False, 
        num_cpus=16,
        num_gpus=1,
        log_to_driver=False
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    # 5. Run the training
    analysis = tune.run(
        "PPO",
        name="PPO_SP",
        # Use NoopLogger to prevent the prometheus/socket.gaierror crash
        loggers=[NoopLogger],
        config={
            # System settings
            "num_gpus": 1,
            # 6 workers + 1 local driver + 1 overhead = 8 CPUs
            "num_workers": 14, 
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            
            # RL Environment setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
            },
            
            # Model Architecture
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 20000000, 
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        # 6. CRITICAL: Save to Scratch to avoid the 15GB Home limit
        local_dir=os.path.expanduser("~/scratch/ray_results"),
    )

    # 7. Post-training Analysis
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    
    if best_trial:
        print(f"Best Trial found: {best_trial}")
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print(f"Best Checkpoint: {best_checkpoint}")
    else:
        print("Training interrupted before a best trial could be determined.")

    print("Process Complete.")
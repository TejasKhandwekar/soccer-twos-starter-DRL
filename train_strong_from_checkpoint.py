import logging
import os
import pickle

import ray
from ray import tune
from ray.tune.logger import NoopLogger

from utils import create_rllib_env


TRAINING_HOURS = int(os.environ.get("STRONG_TRAIN_HOURS", "16"))
TIMESTEP_TARGET = int(os.environ.get("STRONG_TRAIN_TIMESTEPS", "90000000"))
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


def _load_params_from_restore(checkpoint_path: str) -> dict:
    ckpt_dir = os.path.dirname(checkpoint_path)
    params_path = os.path.join(ckpt_dir, "params.pkl")
    if not os.path.exists(params_path):
        params_path = os.path.join(ckpt_dir, "..", "params.pkl")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find params.pkl near checkpoint: {checkpoint_path}")

    with open(params_path, "rb") as f:
        cfg = pickle.load(f)

    return cfg


if __name__ == "__main__":
    if not RESTORE_CHECKPOINT:
        raise ValueError("STRONG_RESTORE_CHECKPOINT must be set for train_strong_from_checkpoint.py")

    os.system("ray stop --force")
    print(f"[Config] STRONG_BASE_PORT={BASE_PORT}")
    print(f"[Config] STRONG_RESTORE_CHECKPOINT={RESTORE_CHECKPOINT}")

    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=16,
        num_gpus=1,
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    config = _load_params_from_restore(RESTORE_CHECKPOINT)

    # Keep original model/policy setup from checkpoint, only force safe runtime knobs.
    config["framework"] = "torch"
    config["log_level"] = "INFO"
    config["num_gpus"] = 1
    config["num_workers"] = int(config.get("num_workers", 14))
    config["num_envs_per_worker"] = int(config.get("num_envs_per_worker", 2))

    env_config = dict(config.get("env_config", {}))
    env_config["base_port"] = BASE_PORT
    env_config["num_envs_per_worker"] = config["num_envs_per_worker"]
    config["env_config"] = env_config
    config["env"] = "Soccer"

    # Slightly lower entropy for better exploitation late in training.
    if "entropy_coeff" in config:
        config["entropy_coeff"] = min(float(config["entropy_coeff"]), 0.001)

    stop_config = {
        "timesteps_total": TIMESTEP_TARGET,
    }

    tune.run(
        "PPO",
        name="PPO_STRONG_RESUME",
        loggers=[NoopLogger],
        restore=RESTORE_CHECKPOINT,
        config=config,
        stop=stop_config,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("~/scratch/ray_results"),
    )

    print("Strong checkpoint resume training complete.")

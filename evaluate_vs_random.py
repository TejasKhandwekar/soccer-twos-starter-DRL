import os
import logging
import argparse
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType
from utils import create_rllib_env

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


def make_opponent(env, mode):
    if mode == "still":
        return lambda *_: 0
    elif mode == "random":
        return lambda *_: env.action_space.sample()
    else:
        raise ValueError(f"Unsupported opponent mode: {mode}")


def evaluate_checkpoint(checkpoint_path, opponent_mode="random", num_episodes=10):
    ray.shutdown()
    ray.init(include_dashboard=False, log_to_driver=False, num_cpus=2)

    tune.registry.register_env("Soccer", create_rllib_env)

    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": True,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }

    agent = PPOTrainer(
        env="Soccer",
        config={
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "env": "Soccer",
            "env_config": env_config,
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        }
    )

    agent.restore(checkpoint_path)

    env = create_rllib_env(env_config)
    env.set_opponent_policy(make_opponent(env, opponent_mode))

    wins, losses, draws = 0, 0, 0
    all_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        final_info = None

        while not done:
            action = agent.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            final_info = info

        all_rewards.append(ep_reward)

        if ep_reward > 0:
            wins += 1
            result = "WIN"
        elif ep_reward < 0:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"

        print(f"Episode {ep+1}/{num_episodes} | reward={ep_reward:.3f} | {result}")

        if ep == 0:
            print("Final info from first episode:")
            print(final_info)

    print("\n===== SUMMARY =====")
    print(f"Opponent  : {opponent_mode}")
    print(f"Wins      : {wins}")
    print(f"Losses    : {losses}")
    print(f"Draws     : {draws}")
    print(f"Win rate  : {wins/num_episodes:.2%}")
    print(f"Avg reward: {np.mean(all_rewards):.4f}")

    env.close()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Full path to checkpoint"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "still"],
        help="Opponent type"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        opponent_mode=args.opponent,
        num_episodes=args.episodes,
    )
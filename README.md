# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 6. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.

## Final Submission (Team KAT)

- Final agent module: `KAT_AGENT/`
- Final packaged artifact for submission: `KAT_AGENT.zip`
- Agent interface implementation: `KAT_AGENT/agent.py` (`StrongPPOAgent`)
- Submission metadata: `KAT_AGENT/README.md`

This repository may contain additional experiment agents and checkpoints, but Team KAT's final submission is the `KAT_AGENT` package.

## Reward Shaping: What and Where

Reward shaping is applied during training (not inference packaging).

- Location of shaping wrapper: `utils.py` (`BallProgressRewardWrapper`)
- Where it is enabled for strong self-play training: `train_strong_selfplay_shaped.py` (`env_config` with `use_ball_progress_reward=True` and `ball_progress_reward_config`)

Dense reward shaping components used for the final KAT strong agent training:

1. Ball progress toward opponent goal (`progress_weight = 0.08`)
2. Offensive territory bonus (`territory_weight = 0.01`)
3. Possession/engagement incentive (`possession_weight = 0.02`)
4. Defensive pressure term (`defense_weight = 0.01`)
5. Conceding penalty (`concede_penalty = 1.0`)

Per-step shaping bonus is clipped to `[-0.20, 0.20]` before being added to base environment reward.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.

## Project Additions (DRL Team Workflow)

This repository now includes a full strong-agent training and evaluation pipeline built on PPO self-play.

### New Training Script

- train_strong_selfplay_shaped.py
	- Multi-agent PPO self-play with opponent archive rotation.
	- Dense reward shaping based on ball progress toward goal.
	- Checkpoint resume support using STRONG_RESTORE_CHECKPOINT.
	- Cluster-safe port handling using STRONG_BASE_PORT.

### New Utility and Packaging/Eval Scripts

- utils.py
	- Includes BallProgressRewardWrapper used by the strong training script.
- package_my_strong_agent.py
	- Finds latest checkpoint and packages files to my_strong_agent/checkpoint.
- evaluate_vs_baseline.py
	- Runs head-to-head module evaluation and saves JSON output.
- evaluate_vs_random.py
	- Runs checkpoint-vs-random evaluation utility.

### Added Batch Scripts (PACE)

- scripts/train_package_eval_strong_pace.sbatch
- scripts/train_strong_agent_pace.sbatch
- scripts/eval_mystrong_vs_baseline_pace.sbatch

These scripts are intended for Georgia Tech PACE usage and include environment setup and job resource configuration.

## Reproducible Strong-Agent Workflow

### 1) Submit/Resume Training on PACE

Example fresh run:

```bash
sbatch scripts/train_package_eval_strong_pace.sbatch
```

Example resume from checkpoint:

```bash
sbatch --export=ALL,STRONG_RESTORE_CHECKPOINT=/path/to/checkpoint-XXXX scripts/train_package_eval_strong_pace.sbatch
```

### 2) Package Latest Checkpoint

```bash
python package_my_strong_agent.py --experiment-dir ~/scratch/ray_results/PPO_STRONG_SELFPLAY_SHAPED
```

### 3) Evaluate Against Baseline

```bash
python evaluate_vs_baseline.py --agent1 my_strong_agent --agent2 ceia_baseline_agent --episodes 200
```

### 4) (Optional) Evaluate Against Random

```bash
python evaluate_vs_random.py --checkpoint /full/path/to/checkpoint-XXXX --opponent random --episodes 20
```

## Collaboration Notes

- Main project remote can point to your team repository.
- Keep upstream linked to the original starter repository for future sync.
- Recommended team flow:
	- Create feature branches per change.
	- Open pull requests into main.
	- Keep experiment outputs and cluster logs out of git.

## Notes on Runtime Warnings

On login nodes, Ray may emit periodic dashboard/metrics warnings related to host resolution.
If evaluation reaches 100 percent episode completion and writes output JSON, results are still valid.

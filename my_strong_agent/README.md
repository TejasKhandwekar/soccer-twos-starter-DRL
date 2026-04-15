# Agent Name
Strong PPO Self-Play + Reward Shaping

# Authors
Tejas Khandwekar

# Emails
your_gt_email@gatech.edu

# Description
This agent package contains a PPO policy trained for SoccerTwos using multi-agent self-play with an opponent archive and dense ball-progress reward shaping.

The `StrongPPOAgent` class implements `soccer_twos.AgentInterface` and returns actions for both controlled players through `act`.

Checkpoint loading options:
- Preferred for packaged submission: place files at `my_strong_agent/checkpoint/checkpoint` and `my_strong_agent/checkpoint/params.pkl`.
- Optional for local testing: set `STRONG_AGENT_CHECKPOINT` to any RLlib checkpoint path.

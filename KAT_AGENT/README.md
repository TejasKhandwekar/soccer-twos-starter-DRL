# Agent Name
Strong PPO Self-Play + Reward Shaping

# Team
KAT

# Authors
Tejas Khandwekar
Akash Misra
Kriti Agrawal

# Emails
tkhandwekar3@gatech.edu
kagrawal74@gatech.edu
amisra43@gatech.edu

# Description
This submission package contains an inference-ready PPO agent for SoccerTwos.

Submission identity note:
- `KAT_AGENT` is the final submission name for Team KAT.
- This is the same Strong PPO agent previously developed as `my_strong_agent`, renamed for submission packaging and naming compliance.
- Core policy class and runtime behavior are unchanged (`StrongPPOAgent` in `agent.py`).

- Interface: `StrongPPOAgent` inherits from `soccer_twos.AgentInterface` and implements `act`.
- Runtime files included: `__init__.py`, `agent.py`, and checkpoint files under `checkpoint/`.
- Training approach: multi-agent self-play with opponent archive and dense reward shaping.

# Reward Shaping Note
The packaged agent is inference-only for autograder runtime, but the checkpoint was trained with dense reward shaping in the project training codebase.

Dense reward shaping used during training:

1. Ball progress toward opponent goal (`progress_weight = 0.08`)
- Rewards step-to-step ball movement in the attacking direction.

2. Offensive territory bonus (`territory_weight = 0.01`)
- Rewards keeping the ball in the opponent half.

3. Possession/engagement incentive (`possession_weight = 0.02`)
- Encourages active ball interaction, approximated via ball movement.

4. Defensive pressure term (`defense_weight = 0.01`)
- Applies a light penalty when ball state indicates defensive vulnerability.

5. Conceding penalty (`concede_penalty = 1.0`)
- Applies a strong negative signal when the opponent scores.

Total shaping reward is clipped to `[-0.20, 0.20]` per step before being added to base environment reward.


Important:
- The checkpoint was trained with reward shaping in the training codebase.
- This submission package is inference-only and does not include training-time reward wrappers.

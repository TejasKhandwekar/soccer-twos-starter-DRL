# Agent Name
PPO Still-Opponent Agent

# Authors
Tejas Khandwekar

# Emails
your_gt_email@gatech.edu

# Description
This agent uses a PPO policy trained in the Soccer-Twos environment using RLlib with a Torch backend. The submitted package wraps a trained checkpoint inside a `soccer_twos.AgentInterface` agent so it can be evaluated with the standard Soccer-Twos rollout tool.

This version corresponds to the PPO agent trained from the `example_ray_ppo_sp_still.py` setup. The training configuration used a 512-unit fully connected policy network with shared value-function layers. The checkpoint included in this package is loaded at inference time, and the agent returns actions for each controlled player through the required `act` method. :contentReference[oaicite:0]{index=0}

This agent package is intended for assignment submission and compatibility with the official evaluation flow. The project instructions require each submitted agent package to include the agent name, authors and emails, and a short description in `README.md`, along with a valid `AgentInterface` implementation and a zipped module folder. 

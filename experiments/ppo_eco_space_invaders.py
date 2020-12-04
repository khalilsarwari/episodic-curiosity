from dotmap import DotMap
from stable_baselines3 import PPO_ECO

# Configuration for PPO with Episodic Curiosity on SpaceInvaders

config = DotMap()
config.agent = PPO_ECO
config.environment = 'SpaceInvaders-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_eco_space_invaders"
config.total_timesteps = 1e7

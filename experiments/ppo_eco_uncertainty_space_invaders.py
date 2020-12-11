from dotmap import DotMap
from stable_baselines3 import PPO_ECO

# Configuration for PPO with Episodic Curiosity on SpaceInvaders

config = DotMap()
config.agent = PPO_ECO
config.environment = 'SpaceInvaders-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_eco_uncertainty_space_invaders"
config.total_timesteps = 2e6
config.ensemble_size = 8
config.rnet_lr = 1e-4
config.atari_wrapper = True
config.add_stoch = True
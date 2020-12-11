from dotmap import DotMap
from stable_baselines3 import PPO

# Configuration for PPO on SpaceInvaders

config = DotMap()
config.agent = PPO
config.environment = 'SpaceInvaders-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_space_invaders"
config.total_timesteps = 2e6
config.atari_wrapper = True
config.add_stoch = True
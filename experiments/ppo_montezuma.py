from dotmap import DotMap
from stable_baselines3 import PPO

# Configuration for PPO on Montezuma's Revenge

config = DotMap()
config.agent = PPO
config.environment = 'MontezumaRevenge-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_montezuma"
config.total_timesteps = 2e6
config.atari_wrapper = True
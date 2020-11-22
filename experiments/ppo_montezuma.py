from dotmap import DotMap
from stable_baselines3 import PPO

# Configuration for PPO on Montezuma's Revenge

config = DotMap()
config.agent = PPO
config.environment = 'MontezumaRevenge-v0'
config.policy_model = 'MlpPolicy'
config.tb_subdir = "ppo_montezuma"
config.total_timesteps = 10000000
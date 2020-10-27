from dotmap import DotMap
from stable_baselines3 import PPO_ECO

# Configuration for PPO with Episodic Curiosity on Montezuma's Revenge

config = DotMap()
config.agent = PPO_ECO
config.environment = 'MontezumaRevenge-v0'
config.policy_model = 'MlpPolicy'
config.tb_subdir = "ppo_eco_montezuma"
config.total_timesteps = 100000

config.reachability = DotMap()
config.reachability.encode_dim = 32
config.reachability.buffer_capacity = 32

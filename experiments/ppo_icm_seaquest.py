from dotmap import DotMap
from stable_baselines3 import PPO_ICM

# Configuration for PPO with Intrinsic Curiosity on Seaquest

config = DotMap()
config.agent = PPO_ICM
config.environment = 'Seaquest-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_icm_seaquest"
config.total_timesteps = 2e6
config.action_shape = 18
config.ensemble_size = 1
config.atari_wrapper = True
config.add_stoch = False

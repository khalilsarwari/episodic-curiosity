import argparse
import importlib
import os
import json
from dotmap import DotMap
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger

def main(config):
    train_env = make_vec_env(config.environment, n_envs=config.workers)

    tb_dir = os.path.join(config.log_dir, config.tb_subdir)
    model = config.agent(config.policy_model, train_env, verbose=config.verbose, tensorboard_log=tb_dir)
    model.config = config.toDict() # save config with tb log

    model.learn(total_timesteps=config.total_timesteps)

    if config.final_vis:
        env = gym.make(config.environment)
        obs = env.reset()
        for i in range(config.final_vis_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="episodic-curiosity")
    parser.add_argument("-exp", "--experiment", type=str, required=True, help='name of config file in experiment folder')
    # per run args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--final_vis", type=bool, default=False)
    parser.add_argument("--final_vis_steps", type=int, default=1000)
    
    args = parser.parse_args()
    config = importlib.import_module('experiments.{}'.format(args.experiment)).config

    # update config dotmap
    config = config.toDict()
    config.update(vars(args))
    config = DotMap(config)

    main(config)
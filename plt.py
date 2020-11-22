import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def plot_mean_and_std(mean, std, line_name, title, xaxis='Iterations', yaxis='Extrinsic Reward per Episode', yrange=None):
    t = list(range(len(mean)))
    plt.plot(t,mean)
    plt.fill_between(t, (mean-std), (mean+std), color='b', alpha=.1)
    plt.savefig(f'{title}_plot.png')

def get_eval_results(file, mean_field, std_field):
    eval_means = []
    eval_stds = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == mean_field:
                eval_means.append(v.simple_value)
            elif v.tag == std_field:
                eval_stds.append(v.simple_value)

    return np.array(eval_means, dtype=np.float32), np.array(eval_stds, dtype=np.float32)

rew_means, rew_stds = get_eval_results('logs/ppo_eco_breakout/PPO_ECO_3/events.out.tfevents.1606075072.AMS4.8895.0', 'rollout/ep_rew_mean', 'rollout/ep_rew_std')
plot_mean_and_std(rew_means, rew_stds, line_name="Reward", title="Breakout PPO EC")


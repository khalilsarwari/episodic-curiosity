import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict


def plot_mean_and_std(mean, std, line_name, title, xaxis='Iterations', yaxis='Extrinsic Reward per Episode', yrange=None):
    """ Plot a mean field and its associated std """
    mean = np.array(mean)
    std = np.array(std)
    t = list(range(len(mean)))
    plt.plot(t,mean)
    plt.fill_between(t, (mean-std), (mean+std), color='b', alpha=.1)
    plt.savefig(f'{title}_plot.png')

def plot_result(result, line_name, title, xaxis='Iterations', yaxis='Extrinsic Reward per Episode', yrange=None):
    """ Plot a regular/simple field, a single line """
    t = list(range(len(result)))
    plt.plot(t,result)
    plt.savefig(f'{title}_plot.png')

def get_results(file, fields):
    """ Gets the fields from the tb log """
    results = defaultdict(lambda:[])
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in fields:
                results[v.tag].append(v.simple_value)

    return results

if __name__ == "__main__":
    results = get_results('logs/ppo_eco_breakout/PPO_ECO_3/events.out.tfevents.1606075072.AMS4.8895.0', ['rollout/ep_rew_mean', 'rollout/ep_rew_std'])
    plot_mean_and_std(results['rollout/ep_rew_mean'], results['rollout/ep_rew_std'], line_name="Reward", title="Breakout PPO EC")
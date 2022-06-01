import argparse

import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading results")
parser.add_argument("--dirname_for_loading", type=str, default=None, help="directory name for loading results")
parser.add_argument("--train_max_episode_len", type=int, default=200, help="length of an episode")
parser.add_argument("--path_for_saving_fig", type=str, default=None)
parser.add_argument("--fig_name", type=str, default=None)
parser.add_argument("--path_for_font", type=str, default=None)
args = parser.parse_args()

### FONT FOR MATPLOTLIB ###
fe = fm.FontEntry(
    fname=os.path.join(args.path_for_font, './ttf/cmunbx.ttf'),
    name='cmunbx')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
matplotlib.rcParams['font.family'] = fe.name # = 'your custom ttf font name'

plt.rc('font', size=20)
### END FONT FOR MATPLOTLIB ###


def runningMean(x, m):
    # Give the sliding mean for the array x over m points.
    y = np.zeros((len(x),))
    window = int((m+1)/2)
    for ctr in range(len(x)):
        if ctr-window < 0:
            y[ctr] = sum(x[0:2*ctr+1]) / len(x[0:2*ctr+1])
        elif ctr+window > len(x):
            y[ctr] = sum(x[2*ctr-len(x)+1:len(x)])/len(x[2*ctr-len(x)+1:len(x)])
        else:
            y[ctr] = sum(x[(ctr-window):(ctr+window) + 1])/len(x[(ctr-window):(ctr+window) + 1])
    return y

def get_success_rate(nb_of_steps_train, train_max_episode_len):
    episode_success = nb_of_steps_train<train_max_episode_len
    nb_of_successes = 0.
    for i in range(len(episode_success)):
        if episode_success[i] == True:
            nb_of_successes += 1.
    success_rate = nb_of_successes/len(episode_success)
    return success_rate

def plot_success_rate(training_file_path, train_max_episode_len):
    nb_of_steps_train = np.loadtxt(training_file_path)
    success_rate = []
    for i in range(nb_of_steps_train.shape[0] - 100):
        success_rate.append(get_success_rate(nb_of_steps_train[i:i+100], train_max_episode_len))
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot()
    plt.plot(runningMean(success_rate, 100), '-', linewidth=3, color='tab:blue')
    axes.set_xlabel('Number of episodes', fontsize=34)
    axes.set_ylabel('Success rate', fontsize=34)
    for tick in axes.xaxis.get_ticklabels():
        tick.set_weight('bold')
    for tick in axes.yaxis.get_ticklabels():
        tick.set_weight('bold')
    axes.tick_params(axis='both', which='major', labelsize=28, width=1)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes.xaxis.get_offset_text().set_visible(True)
    plt.rc('axes', unicode_minus=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.path_for_saving_fig, './{}.pdf'.format(args.fig_name)), format='pdf')
    plt.show()

plot_success_rate(os.path.join(args.path_for_loading, './{}/nb_of_steps.npy'.format(args.dirname_for_loading)), args.train_max_episode_len)
import argparse

import numpy as np
import gym_systmemoire
import gym_systmemoire.envs

import Config_env

import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading positions")
parser.add_argument("--dirname_for_loading", type=str, default=None, help="directory name for loading positions")
parser.add_argument("--env_name", type=str, default='env-3M-v0', help="environment name")
parser.add_argument("--env_surname", type=str, default='3M-v0', help="environment name")
parser.add_argument("--transition", type=str, default='111_to_001')
parser.add_argument("--goal_bin", default=[0, 0, 1], help="target state")
parser.add_argument("--not_goal_bin", default=[1, 1, 0])
parser.add_argument("--c", type=float, default=2.)
parser.add_argument("--num_fig", type=int, default=1)
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

def bin_to_pos(binstate, eq_pos):
    if binstate == 0:
        become = eq_pos[0]
    else:
        become = eq_pos[2]
    return become

def goalbin_to_goalpos(eq_positions, goalbin, notgoalbin, nb_ressort):
    goalpos = np.zeros_like(goalbin, dtype='float')
    notgoalpos = np.zeros_like(notgoalbin, dtype='float')
    for i in range(nb_ressort):
        goalpos[i] = bin_to_pos(goalbin[i], eq_positions[i])
        notgoalpos[i] = bin_to_pos(notgoalbin[i], eq_positions[i])
    return goalpos.tolist(), notgoalpos.tolist()

def get_elongation(positions):
    # Get elongations
    rel_pos = np.diff(positions, axis=0)

    for i in range(positions[0].shape[0] - 1, -1, -1):
        rel_pos = np.append(positions[0][i], rel_pos)
    rel_pos = rel_pos.reshape(-1, positions[0].shape[0])
    return rel_pos


positions = np.loadtxt(os.path.join(args.path_for_loading, './{}/pos_{}_transition_{}_{}.npy'.format(args.dirname_for_loading, 'c_{}_{}'.format(int(args.c), int(str(args.c).split('.')[1])), args.transition, args.num_fig)))

nb_ressort = np.size(Config_env.exp[args.env_surname]['system'])
eq_positions = np.array([Config_env.exp[args.env_surname]['system'][k].x_e for k in range(nb_ressort)])
goal, notgoal = goalbin_to_goalpos(eq_positions, args.goal_bin, args.not_goal_bin, nb_ressort)


show_y = []

cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
if nb_ressort == 1:
    steps = np.linspace(0, positions.shape[0], positions.shape[0])

    goal_plt = np.empty((len(goal), positions.shape[0]))
    notgoal_plt = np.empty((len(notgoal), positions.shape[0]))

    for i in range(len(goal)):
        goal_plt[i] = [goal[i] for k in range(positions.shape[0])]
        notgoal_plt[i] = [notgoal[i] for k in range(positions.shape[0])]

    fig, axes = plt.subplots(figsize=(9., 6.))
    axes.plot(steps, positions, color=cycle[0], linewidth=3)
    if args.goal_bin == 0:
        axes.plot(steps, goal_plt[0], '--', color='#dd4b39')
        axes.plot(steps, notgoal_plt[0], '--', color='#162347')
    else:
        axes.plot(steps, goal_plt[0], '--', color='#162347')
        axes.plot(steps, notgoal_plt[0], '--', color='#dd4b39')
    axes.set_yticks(show_y)
    for tick in axes.xaxis.get_ticklabels():
        tick.set_weight('bold')
    for tick in axes.yaxis.get_ticklabels():
        tick.set_weight('bold')
    axes.tick_params(axis='both', which='major', labelsize=28, width=1)
    plt.rc('axes', unicode_minus=False)
    fig.tight_layout()
    plt.savefig(os.path.join(args.path_for_saving_fig, './elongations_{}__{}.pdf'.format(args.transition, args.fig_name)), format='pdf')

else:
    elongations = get_elongation(positions)
    steps = np.linspace(0, positions.shape[1], positions.shape[1])

    goal_plt = np.empty((len(goal), positions.shape[1]))
    notgoal_plt = np.empty((len(notgoal), positions.shape[1]))

    for i in range(len(goal)):
        goal_plt[i] = [goal[i] for k in range(positions.shape[1])]
        notgoal_plt[i] = [notgoal[i] for k in range(positions.shape[1])]

    fig, axes = plt.subplots(nrows=nb_ressort, sharex='all', figsize=(9., 6.))
    for i in range(nb_ressort):
        axes[i].plot(steps, elongations[nb_ressort - 1 - i], color=cycle[nb_ressort - 1 - i], linewidth=3)
        if args.goal_bin == 0:
            axes[i].plot(steps, goal_plt[nb_ressort - 1 - i], '--', color='#dd4b39')
            axes[i].plot(steps, notgoal_plt[nb_ressort - 1 - i], '--', color='#162347')
        else:
            axes[i].plot(steps, goal_plt[nb_ressort - 1 - i], '--', color='#162347')
            axes[i].plot(steps, notgoal_plt[nb_ressort - 1 - i], '--', color='#dd4b39')
        axes[i].set_yticks(show_y)
    for _ in range(nb_ressort):
        for tick in axes[_].xaxis.get_ticklabels():
            tick.set_weight('bold')
        for tick in axes[_].yaxis.get_ticklabels():
            tick.set_weight('bold')
        axes[_].tick_params(axis='both', which='major', labelsize=28, width=1)
    plt.rc('axes', unicode_minus=False)
    fig.tight_layout()
    plt.savefig(os.path.join(args.path_for_saving_fig, './elongations_{}__{}.pdf'.format(args.transition, args.fig_name)), format='pdf')


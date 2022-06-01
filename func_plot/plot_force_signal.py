import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading force signal")
parser.add_argument("--dirname_for_loading", type=str, default=None, help="directory name for loading force signal")
parser.add_argument("--env_name", type=str, default='env-3M-v0', help="environment name")
parser.add_argument("--env_surname", type=str, default='3M-v0', help="environment name")
parser.add_argument("--transition", type=str, default='111_to_001')
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


force = np.loadtxt(os.path.join(args.path_for_loading, './{}/force_{}_transition_{}_{}.npy'.format(args.dirname_for_loading, 'c_{}_{}'.format(int(args.c), int(str(args.c).split('.')[1])), args.transition, args.num_fig)))
steps = np.linspace(0, force.shape[0], force.shape[0])

show = []

fig = plt.figure(figsize=(9., 6))
axes = fig.add_subplot()
plt.plot(steps, force, '-', linewidth=3, markersize=18, color='tab:blue')
for tick in axes.xaxis.get_ticklabels():
    tick.set_weight('bold')
for tick in axes.yaxis.get_ticklabels():
    tick.set_weight('bold')
plt.rc('axes', unicode_minus=False)
axes.tick_params(axis='both', which='major', labelsize=28, width=1)
plt.savefig(os.path.join(args.path_for_saving_fig, './force_{}__{}.pdf'.format(args.transition, args.fig_name)), format='pdf')

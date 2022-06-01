import argparse

import numpy as np

from Functions import get_stop

import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading results and models")
parser.add_argument("--dirname_for_loading_TL", type=str, default='c_{}_{}', help="directory name for loading results with TL")
parser.add_argument("--dirname_for_loading", type=str, default=None, help="directory name for loading results without TL")
parser.add_argument("--threshold", typr=float, default=0.8, help="threshold to define learning time")
parser.add_argument("--c_gridsearch_transfer", type=np.array, default=[0., 0.1, 0.2, 0.3, 0.6, 1., 3., 4., 5., 6., 7., 8., 9., 10.], help="dissipation gridsearch for TL")
parser.add_argument("--c_gridsearch", type=np.array, default=[0., 0.1, 0.2, 0.3, 0.6, 1., 2., 3., 4., 5., 6., 7.], help="dissipation gridsearch")
parser.add_argument("--path_for_saving_fig", tyoe=str, default=None)
parser.add_argument("--fig_name", type=str, default='Transfer_learning')
parser.add_argument("--path_for_font", typr=str, default=None)
args = parser.parse_args()

### FONT FOR MATPLOTLIB ###
fe = fm.FontEntry(
    fname=os.path.join(args.path_for_font, './ttf/cmunbx.ttf'),
    name='cmunbx')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
matplotlib.rcParams['font.family'] = fe.name # = 'your custom ttf font name'

plt.rc('font', size=20)
### END FONT FOR MATPLOTLIB ###

learning_time_transfer = []
for c in args.c_gridsearch_transfer:
    learning_time_transfer.append(get_stop(os.path.join(args.path_for_loading, './{}'.format(args.dirname_for_loading_TL.format(int(c), int(str(c).split('.')[1])))), args.threshold))

learning_time = []
for c in args.c_gridsearch:
    learning_time.append(get_stop(os.path.join(args.path_for_loading, './{}'.format(args.dirname_for_loading.format(int(c), int(str(c).split('.')[1])))), args.threshold))


show = [0., 2., 4., 6., 8., 10]

fig = plt.figure(figsize=(8, 6))
axes = fig.add_subplot()
plt.plot(args.c_gridsearch_transfer, learning_time_transfer, '^', color='k', markersize=12, label='with TL')
plt.plot(args.c_gridsearch, learning_time, '.', color='k', markersize=18, label='wihtout TL')
axes.set_xticks(show)
axes.set_xlabel('{} (Kg/s)'.format(chr(951)), fontsize=34)
axes.set_ylabel('Learning time', fontsize=34)
for tick in axes.xaxis.get_ticklabels():
    tick.set_weight('bold')
for tick in axes.yaxis.get_ticklabels():
    tick.set_weight('bold')
axes.tick_params(axis='both', which='major', labelsize=28, width=1)
plt.legend(prop={'size': 28}, loc='upper left')
axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.savefig(os.path.join(args.path_for_saving_fig, './{}.pdf'.format(args.fig_name)), format='pdf')
plt.show()



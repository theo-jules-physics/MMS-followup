import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm

parser = argparse.ArgumentParser()
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading results")
parser.add_argument("--dirname_for_loading", type=str, default='m_{}_{}_Fmax_{}_{}_c_{}_{}_{}', help="directory for loading results")
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

#The parameters we used are the following
m_gridsearch = np.array([0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.5, 2., 3., 5., 7., 10., 20.])  # 100.
Fmax_gridsearch = np.array([1.])
dict_c_gridsearch_m = {'0.01': np.array([0., 0.1, 0.2, 0.3, 0.6, 1., 1.5, 2.]),
                       '0.03': np.array([0., 0.3, 0.6, 1., 2., 3., 4.]),
                       '0.05': np.array([0., 0.1, 0.2, 0.3, 0.6, 1., 1.5, 2.]),
                       '0.07': np.array([0., 0.3, 0.6, 1., 1.5, 2., 3., 4.]),
                       '0.2': np.array([0., 0.6, 1., 2., 3., 4., 5.]),
                       '0.1': np.array([0., 0.6, 1., 1.5, 2., 2.5, 3., 4., 5.]),
                       '0.3': np.array([0., 0.6, 1., 2., 3., 4., 5.]),
                       '0.5': np.array([0., 0.6, 1., 1.5, 2., 3., 4., 5.]),
                       '0.8': np.array([0., 1., 1.5, 2., 3., 4., 5.]),
                       '1.0': np.array([0., 0.6, 1., 2., 2.5, 3., 4.]),
                       '1.5': np.array([0., 1., 1.5, 2., 2.5, 3., 4., 5.]),
                       '2.0': np.array([0., 1., 2., 3., 4., 5., 6.]),
                       '3.0': np.array([0., 1., 2., 3., 4., 5., 6.]),
                       '5.0': np.array([0., 1., 2., 3., 4., 5., 6.]),
                       '7.0': np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
                       '10.0': np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
                       '20.0': np.array([2., 4., 6., 7., 8., 9., 10., 11.]),
                       '100.0': np.array([0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 21., 22., 24., 26.])}
dict_transitions_gridsearch_m = {'0.01': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                   '101_from_111', '011_from_111', '000_from_111', '001_from_111',
                                                   '011_from_000', '010_from_111', '011_from_101', '110_from_010',
                                                   '011_from_100', '001_from_100']),
                                 '0.03': np.array(['110_from_000', '101_from_000', '111_from_000']),
                                 '0.05': np.array(['111_from_000', '010_from_000', '110_from_000', '101_from_000',
                                                   '101_from_111', '001_from_111', '000_from_111']),
                                 '0.07': np.array(['010_from_000', '110_from_000', '101_from_000',
                                                   '111_from_000', '101_from_111']),
                                 '0.1': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111',
                                                  '011_from_000', '010_from_111', '001_from_000', '011_from_101',
                                                  '110_from_010', '001_from_100', '011_from_100']),
                                 '0.2': np.array(['110_from_000', '101_from_000', '111_from_000',
                                                  '011_from_111', '000_from_111', '001_from_111']),
                                 '0.3': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '0.5': np.array(['010_from_000', '110_from_000', '111_from_000', '101_from_111',
                                                  '011_from_111', '000_from_111', '001_from_111']),
                                 '0.8': np.array(['110_from_000', '101_from_000', '111_from_000',
                                                  '011_from_111', '000_from_111', '001_from_111']),
                                 '1.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111',
                                                  '011_from_000', '010_from_111', '011_from_101', '110_from_010',
                                                  '001_from_100', '011_from_100']),
                                 '1.5': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '2.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '3.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '5.0': np.array(['110_from_000', '101_from_000', '000_from_111', '001_from_111',]),
                                 '7.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111']),
                                 '10.0': np.array(['010_from_000', '101_from_111', '011_from_000', '111_from_110',
                                                   '110_from_000', '111_from_000']),
                                 '20.0': np.array(['010_from_000', '101_from_111', '011_from_000', '110_from_000'])}

m_gridsearch_force = np.array([1.])
Fmax_gridsearch_force = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.5, 2., 3., 5., 7., 10., 15.])  # 100.
dict_transitions_gridsearch_F = {'0.01': np.array(['010_from_000', '101_from_000', '101_from_111', '011_from_000',
                                                   '001_from_000', '111_from_110', '100_from_111', '110_from_111',
                                                   '111_from_100']),
                                 '0.03': np.array(['010_from_000', '101_from_000', '111_from_000']),
                                 '0.05': np.array(['101_from_111', '011_from_000', '111_from_110', '000_from_101',
                                                   '111_from_000']),
                                 '0.07': np.array(['010_from_000', '110_from_000', '101_from_000']),
                                 '0.1': np.array(['011_from_000', '110_from_010', '001_from_100', '011_from_100',
                                                  '100_from_111', '111_from_100']),
                                 '0.2': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '0.3': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '0.5': np.array(['110_from_000', '101_from_000', '111_from_000', '101_from_111',
                                                  '011_from_111', '000_from_111', '001_from_111', '010_from_000']),
                                 '0.8': np.array(['110_from_000', '101_from_000', '111_from_000', '011_from_111',
                                                  '000_from_111', '001_from_111']),
                                 '1.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111',
                                                  '011_from_000', '010_from_111', '011_from_101', '110_from_010',
                                                  '001_from_100', '011_from_100']),
                                 '1.5': np.array(['110_from_000', '101_from_000', '111_from_000', '000_from_111',
                                                  '001_from_111']),
                                 '2.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '5.0': np.array(['110_from_000', '111_from_000', '011_from_111', '000_from_111',
                                                  '001_from_111', '101_from_000']),
                                 '3.0': np.array(['010_from_000', '110_from_000', '101_from_000', '111_from_000',
                                                  '101_from_111', '011_from_111', '000_from_111', '001_from_111']),
                                 '7.0': np.array(['110_from_000', '101_from_000', '111_from_000', '011_from_111',
                                                  '001_from_111']),
                                 '10.0': np.array(['101_from_111', '011_from_111', '000_from_111', '001_from_111',
                                                   '010_from_111', '110_from_010', '001_from_100', '011_from_100',
                                                   '100_from_111', '011_from_101', '110_from_000', '011_from_000',
                                                   '000_from_101'
                                                   ]),
                                 '15.0': np.array(['010_from_000', '110_from_000', '111_from_000'])}

dict_c_gridsearch_F = {'0.01': np.array([0., 0.1, 0.2, 0.3, 0.6, 1., 1.5, 2., 2.5, 3.]),
                       '0.03': np.array([0., 0.3, 0.6, 1., 2., 3., 4.]),
                       '0.05': np.array([0., 0.1, 0.2, 0.3, 0.6, 1., 1.5, 2., 2.5, 3.]),
                       '0.07': np.array([0., 0.3, 0.6, 1., 1.5, 2., 3., 4.]),
                       '0.2': np.array([0., 0.6, 1., 1.5, 2., 3., 4., 5.]),
                       '0.3': np.array([0., 0.6, 1., 2., 3., 4., 5.]),
                       '0.5': np.array([0., 0.6, 1., 1.5, 2., 2.5, 3.]),
                       '0.1': np.array([0., 0.6, 1., 1.5, 2., 2.5, 3., 4., 5.]),
                       '0.8': np.array([0., 1., 1.5, 2., 3., 4., 5.]),
                       '1.0': np.array([0., 0.6, 1., 2., 2.5, 3., 4.]),
                       '1.5': np.array([0., 1., 1.5, 2., 2.5, 3., 4., 5.]),
                       '2.0': np.array([0., 1., 2., 3., 4., 5., 6.]),
                       '3.0': np.array([0., 1., 2., 3., 4., 5., 6.]),
                       '5.0': np.array([0., 1., 2., 2.5, 3., 4., 5., 6.]),
                       '7.0': np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
                       '10.0': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]),
                       '15.0': np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
                       '100.0': np.array([0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.])}


def get_min_nb_of_steps(path):
    nb_of_steps = np.loadtxt(os.path.join(path, './nb_of_steps.npy'))
    min_nb_of_of_steps = np.min(nb_of_steps)
    return min_nb_of_of_steps


# m, Fmax and transition are fixed
# take a pandas DataFrame with min_nb_of_steps associated with each c
def get_c_0(min_nb_of_steps_array, c_array):
    data = pd.DataFrame(data=min_nb_of_steps_array, index=c_array)
    return pd.DataFrame.idxmin(data)


# LOAD FILES AND RETURNS AN ARRAY WITH THE MIN NB OF STEPS FOR EVERY C AT TRANSITION, M, FMAX FIXED
def load(transition, mass, Fmax, dict_c_gridsearch, force):
    if force == True:
        min_nb_of_steps_array = []
        for dissipation in range(dict_c_gridsearch['{}'.format(args.f_gridsearch[Fmax])].shape[0]):
            nb_of_steps = np.loadtxt(os.path.join(args.path_for_loading, './{}/nb_of_steps.npy'.format(args.dirname_for_loading.format(int(m_gridsearch_force[mass]),
                                                                                                       (str(m_gridsearch_force[mass]).split('.')[1]),
                                                                                                       int(Fmax_gridsearch_force[Fmax]),
                                                                                                       (str(Fmax_gridsearch_force[Fmax]).split('.')[1]),
                                                                                                       int(dict_c_gridsearch['{}'.format(Fmax_gridsearch_force[Fmax])][dissipation]),
                                                                                                       (str(dict_c_gridsearch['{}'.format(Fmax_gridsearch_force[Fmax])][dissipation]).split('.')[1]),
                                                                                                       transition))))
            min_nb_of_steps = np.min(nb_of_steps)
            min_nb_of_steps_array.append(min_nb_of_steps)
        # ARRAY OF MIN NUMBER OF STEPS FOR EVERY C AT TRANSITION, FMAX, M FIXED
        min_nb_of_steps_array = np.array(min_nb_of_steps_array)
    else:
        min_nb_of_steps_array = []
        for dissipation in range(dict_c_gridsearch['{}'.format(m_gridsearch[mass])].shape[0]):
            nb_of_steps = np.loadtxt(os.path.join(args.path_for_loading, './{}/nb_of_steps.npy'.format(args.dirname_for_loading.format(
                                     int(m_gridsearch[mass]), (str(m_gridsearch[mass]).split('.')[1]),
                                     int(Fmax_gridsearch[Fmax]), (str(Fmax_gridsearch[Fmax]).split('.')[1]),
                                     int(dict_c_gridsearch['{}'.format(m_gridsearch[mass])][dissipation]),
                                     (str(dict_c_gridsearch['{}'.format(m_gridsearch[mass])][dissipation]).split('.')[1]),
                                     transition))))
            min_nb_of_steps = np.min(nb_of_steps)
            min_nb_of_steps_array.append(min_nb_of_steps)
        # ARRAY OF MIN NUMBER OF STEPS FOR EVERY C AT TRANSITION, FMAX, M FIXED
        min_nb_of_steps_array = np.array(min_nb_of_steps_array)
    return min_nb_of_steps_array


def get_c_0_DataFrame(force=False):
    if force == True:
        c_0_mat = np.empty((Fmax_gridsearch_force.shape[0], m_gridsearch_force.shape[0]))
        for mass in range(m_gridsearch_force.shape[0]):
            for Fmax in range(Fmax_gridsearch_force.shape[0]):
                c_0_list = []
                for transition in dict_transitions_gridsearch_F['{}'.format(Fmax_gridsearch_force[Fmax])]:
                    min_nb_of_steps_array = load(transition, mass, Fmax, dict_c_gridsearch_F, force=True)
                    c_0 = get_c_0(min_nb_of_steps_array, dict_c_gridsearch_F['{}'.format(Fmax_gridsearch_force[Fmax])])
                    c_0_list.append(c_0)
                c_0_mean = np.mean(c_0_list)
                c_0_mat[Fmax, mass] = c_0_mean
        c_0_DataFrame_ = pd.DataFrame(data=c_0_mat, columns=m_gridsearch_force, index=Fmax_gridsearch_force)
    else:
        c_0_mat = np.empty((m_gridsearch.shape[0], Fmax_gridsearch.shape[0]))
        for mass in range(m_gridsearch.shape[0]):
            for Fmax in range(Fmax_gridsearch.shape[0]):
                c_0_list = []
                for transition in dict_transitions_gridsearch_m['{}'.format(m_gridsearch[mass])]:
                    min_nb_of_steps_array = load(transition, mass, Fmax, dict_c_gridsearch_m, force=False)
                    c_0 = get_c_0(min_nb_of_steps_array, dict_c_gridsearch_m['{}'.format(m_gridsearch[mass])])
                    c_0_list.append(c_0)
                c_0_mean = np.mean(c_0_list)
                c_0_mat[mass, Fmax] = c_0_mean
        c_0_DataFrame_ = pd.DataFrame(data=c_0_mat, columns=Fmax_gridsearch, index=m_gridsearch)
    return c_0_DataFrame_


def get_c_0_from_formula(k, m, Fmax):
    return m ** (1. / 2.) * k ** (1. / 6.) * Fmax ** (1. / 3.)


def get_c_0_DataFrame_from_formula(force):
    if force == True:
        c_0_mat = np.empty((Fmax_gridsearch_force.shape[0], m_gridsearch_force.shape[0]))
        for mass in range(m_gridsearch_force.shape[0]):
            for force_max in range(Fmax_gridsearch_force.shape[0]):
                c_0_mat[force_max, mass] = get_c_0_from_formula(88.7, m_gridsearch_force[mass],
                                                                Fmax_gridsearch_force[force_max])
        c_0_DataFrame = pd.DataFrame(data=c_0_mat, columns=m_gridsearch_force, index=Fmax_gridsearch_force)
    else:
        c_0_mat = np.empty((m_gridsearch.shape[0], Fmax_gridsearch.shape[0]))
        for mass in range(m_gridsearch.shape[0]):
            for force_max in range(Fmax_gridsearch.shape[0]):
                c_0_mat[mass, force_max] = get_c_0_from_formula(88.7, m_gridsearch[mass], Fmax_gridsearch[force_max])
        c_0_DataFrame = pd.DataFrame(data=c_0_mat, columns=Fmax_gridsearch, index=m_gridsearch)
    return c_0_DataFrame


c_0_DataFrame_exp = get_c_0_DataFrame()
c_0_DataFrame_theory = get_c_0_DataFrame_from_formula(force=False)

print('exp m : ', c_0_DataFrame_exp)
print('theory m : ', c_0_DataFrame_theory)

m = [k for k in np.linspace(0.01, 25., 1000)]
c_0 = []
for mass in m:
    c_0.append(get_c_0_from_formula(88.7, mass, 1.))

fig = plt.figure(figsize=(8., 6.))
axes = fig.add_subplot()
plt.plot(m_gridsearch, c_0_DataFrame_exp, '.', color='tab:blue', markersize=18)
plt.plot(m, c_0, color='black', linewidth=1)
axes.set_xlabel('m (Kg)', fontsize=34)
axes.set_ylabel('{} (Kg/s)'.format(chr(951)), fontsize=34)
axes.set_xscale('log')
axes.set_yscale('log')
for tick in axes.xaxis.get_ticklabels():
    tick.set_weight('bold')
for tick in axes.yaxis.get_ticklabels():
    tick.set_weight('bold')
plt.rc('axes', unicode_minus=False)
axes.tick_params(axis='both', which='major', labelsize=28, width=1)
#plt.legend(prop={'size': 24}, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(args.path_for_saving_fig, './{}__mass.pdf'.format(args.fig_name)), format='pdf')
plt.show()

c_0_DataFrame_exp_force = get_c_0_DataFrame(force=True)
c_0_DataFrame_theory_force = get_c_0_DataFrame_from_formula(force=True)

print('exp force : ', c_0_DataFrame_exp_force)
print('theory force : ', c_0_DataFrame_theory_force)

F = [k for k in np.linspace(0.01, 20., 1000)]
c_0 = []
for force in F:
    c_0.append(get_c_0_from_formula(88.7, 1., force))


fig = plt.figure(figsize=(8., 6.))
axes = fig.add_subplot()
plt.plot(Fmax_gridsearch_force, c_0_DataFrame_exp_force, '.', color='tab:blue', markersize=18)
plt.plot(F, c_0, color='black', linewidth=1)
axes.set_xlabel('Fmax (N)', fontsize=34)
axes.set_ylabel('{} (Kg/s)'.format(chr(951)), fontsize=34)
axes.set_xscale('log')
axes.set_yscale('log')
for tick in axes.xaxis.get_ticklabels():
    tick.set_weight('bold')
for tick in axes.yaxis.get_ticklabels():
    tick.set_weight('bold')
plt.rc('axes', unicode_minus=False)
axes.tick_params(axis='both', which='major', labelsize=28, width=1)
plt.tight_layout()
plt.savefig(os.path.join(args.path_for_saving_fig, './{}__force_max.pdf'.format(args.fig_name)), format='pdf')
plt.show()

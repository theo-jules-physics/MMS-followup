import argparse
import sys

import numpy as np

from pfrl import experiments

from Classes import Train

import Config_env


parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="results",help=("Directory path to save output files." " If it does not exist, it will be created."),)
parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32]")
parser.add_argument("--put_seed", type=bool, default=True, help="put seeds for reproducibility")
parser.add_argument("--gpu", type=int, default=0, help="GPU name. -1 if no GPU.")
parser.add_argument("--env_name", type=str, default='env-3M-v0', help="environment name")
parser.add_argument("--env_surname", type=str, default='3M-v0', help="environment name")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--rb_size", type=int, default=10**6, help="Replay buffer size")
parser.add_argument("--tau", type=float, default=5e-3, help="soft update tau")
parser.add_argument("--replay_start_size", type=int, default=10000, help="exploration time")
parser.add_argument("--minibatch_size", type=int, default=100, help="batchsize for updates")
parser.add_argument("--steps", type=int, default=2000000, help="total number of steps")
parser.add_argument("--eval_n_episodes", type=int, default=10, help="for tracking training")
parser.add_argument("--eval_interval", type=int, default=2000, help="for tracking training")
parser.add_argument("--train_max_episode_len", type=int, default=200, help="length of an episode")
parser.add_argument("--path_to_save", type=str, default=None, help="path for results and models")
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading models")
parser.add_argument("--dirname_for_loading", type=str, default='c_{}_{}', help="directory for loading models")
parser.add_argument("--dirname_to_save", type=str, default="m_{}_{}_Fmax_{}_{}_c_{}_{}_{}", help="directory to put results in")
parser.add_argument("--tar_act_noise", type=float, default=0.5, help="maximum target actions noise")
parser.add_argument("--c_gridsearch", default=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], help="dissipation gridsearch")
parser.add_argument("--m_gridsearch", default=[0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.5, 2., 3., 5., 7., 10., 20.], help="mass gridsearch")
parser.add_argument("--f_gridsearch", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.5, 2., 3., 5., 7., 10., 15.], help="Maximal amplitude force gridsearch")
parser.add_argument("--transitions_gridsearch", default=['000_to_001', '000_to_010', '000_to_011',
                                                         '000_to_100', '000_to_101', '000_to_110',
                                                         '000_to_111', '001_to_000', '001_to_010',
                                                         '001_to_011', '001_to_100', '001_to_101',
                                                         '001_to_110', '001_to_111', '010_to_000',
                                                         '010_to_001', '010_to_011', '010_to_100',
                                                         '010_to_101', '010_to_110', '010_to_111',
                                                         '011_to_000', '011_to_001', '011_to_010',
                                                         '011_to_100', '011_to_101', '011_to_110',
                                                         '011_to_111', '100_to_000', '100_to_001',
                                                         '100_to_010', '100_to_011', '100_to_101',
                                                         '100_to_110', '100_to_111', '101_to_000',
                                                         '101_to_001', '101_to_010', '101_to_011',
                                                         '101_to_100', '101_to_110', '101_to_111',
                                                         '110_to_000', '110_to_001', '110_to_010',
                                                         '110_to_011', '110_to_100', '110_to_101',
                                                         '110_to_111', '111_to_000', '111_to_001',
                                                         '111_to_010', '111_to_011', '111_to_100',
                                                         '111_to_101', '111_to_110',
                                                        ])
parser.add_argument("--init_gridsearch", default=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                                                  [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
                                                  [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                                                  [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
                                                  [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                                                  [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
parser.add_argument("--target_gridsearch", default=[[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                                     [0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                                     [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                                     [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                                     [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                                                     [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1],
                                                     [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1],
                                                     [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
args = parser.parse_args()


args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
print("Output files are saved in {}".format(args.outdir))


def bin_to_pos(binstate, eq_pos):
    if binstate == 0:
        become = eq_pos[0]
    else:
        become = eq_pos[2]
    return become


def goalbin_to_goalpos(eq_positions, init_state, nb_ressort):
    goalpos = np.zeros_like(init_state, dtype='float')
    goalpos_cum = np.zeros_like(init_state, dtype='float')
    for i in range(nb_ressort):
        goalpos[i] = bin_to_pos(init_state[i], eq_positions[i])
    goalpos_cum[0] = goalpos[0]
    for i in range(nb_ressort - 1):
        goalpos_cum[i + 1] = goalpos_cum[i] + goalpos[i + 1]
    goalpos_cum = np.append(goalpos_cum, np.zeros(nb_ressort))
    return goalpos_cum.tolist()

for mass in range(len(args.m_gridsearch)):
    for force_max in range(len(args.f_gridsearch)):
        for transition in range(len(args.transitions_gridsearch)):
            for dissipation in range(len(args.c_gridsearch)):

                nb_ressort = np.size(Config_env.exp[args.env_surname]['system'])
                Config_env.exp[args.env_surname]['c'] = args.c_gridsearch[dissipation]
                Config_env.exp[args.env_surname]['masse'] = np.array([args.m_gridsearch[mass], args.m_gridsearch[mass], args.m_gridsearch[mass]])
                Config_env.exp[args.env_surname]['max_f'] = args.f_gridsearch[force_max]
                Config_env.exp[args.env_surname]['c'] = args.c_gridsearch[dissipation]
                Config_env.exp[args.env_surname]['goal_state'] = args.target_gridsearch[transition]
                eq_positions = np.array([Config_env.exp[args.env_surname]['system'][k].x_e for k in range(nb_ressort)])

                init_pos = goalbin_to_goalpos(eq_positions, args.init_gridsearch[transition], nb_ressort)
                Config_env.exp[args.env_surname]['ini_pos'] = init_pos
                print('init_pos : ', init_pos)


                train = Train(seed=args.seed,
                              put_seed=args.put_seed,
                              gpu=args.gpu,
                              env_name=args.env_name,
                              gamma=args.gamma,
                              rb_size=args.rb_size,
                              tau=args.tau,
                              replay_start_size=args.replay_start_size,
                              minibatch_size=args.minibatch_size,
                              steps=args.steps,
                              eval_n_episodes=args.eval_n_episodes,
                              eval_interval=args.eval_interval,
                              train_max_episode_len=args.train_max_episode_len,
                              path_to_save=args.path_to_save,
                              path_for_loading=args.path_for_loading,
                              dirname_to_save=args.dirname_to_save.format(int(args.m_gridsearch[mass]), (str(args.m_gridsearch[mass]).split('.')[1]),
                                                                                     int(args.f_gridsearch[force_max]), (str(args.f_gridsearch[force_max]).split('.')[1]),
                                                                                     int(args.c_gridsearch[dissipation]), (str(args.c_gridsearch[dissipation]).split('.')[1]),
                                                                                     args.transitions_gridsearch[transition]),
                              dirname_for_loading=args.dirname_for_loading.format(int(args.c_gridsearch[dissipation]), (str(args.c_gridsearch[dissipation]).split('.')[1])),
                              tar_act_noise=args.tar_act_noise,
                              )

                agent, eval_env = train.main()
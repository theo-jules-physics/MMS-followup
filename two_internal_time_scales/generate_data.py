import argparse
from Classes import Test_phase
import Config_env


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU name. -1 if no GPU.")
parser.add_argument("--env_name", type=str, default='env-3M-v0', help="environment name")
parser.add_argument("--env_surname", type=str, default='3M-v0', help="environment name")
parser.add_argument("--pos_noise", type=float, default=0.001, help="add gaussian noise to the initial position")
parser.add_argument("--vel_noise", type=float, default=0.01, help="start with gaussian noise velocities")
parser.add_argument("--n_steps", type=int, default=None, help="total number of steps. If None, the number of episodes put will be taken into account.")
parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to run.")
parser.add_argument("--max_episode_len", type=int, default=200, help="length of an episode")
parser.add_argument("--add_simu_time", type=bool, default=False, help="add time at the end of an episode, letting the system relax.")
parser.add_argument("--t_simu", type=int, default=None, help="length of the added time")
parser.add_argument("--file_name", type=str, default='c_{}_{}_transition_{}', help="file name for pos/vel/force results")
parser.add_argument("--path_to_save", type=str, default=None, help="path for pos/vel/force results")
parser.add_argument("--dirname_to_save", type=str, default=None, help="dirname to save pos/vel/force results")
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading models")
parser.add_argument("--dirname_for_loading", type=str, default='c_{}_{}', help="dirname for loading models")
parser.add_argument("--c_gridsearch", type=float, default=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], help="value of the dissipation coefficient")
parser.add_argument("--init_state", default=[1, 1, 1])
parser.add_argument("--target_state", default=[0, 0, 1])
parser.add_argument("--transitions_gridsearch", default=['111_to_001'])
parser.add_argument("--save_obs", type=bool, default=True, help="save pos/vel/force results")
args = parser.parse_args()

for transition in range(len(args.transitions_gridsearch)):
    for c in args.c_gridsearch:

        if args.pos_noise is not None:
            Config_env.exp[args.env_surname]['add_gaussian_noise_to_ini_pos'] = True
            Config_env.exp[args.env_surname]['pos_noise'] = args.pos_noise
        if args.vel_noise is not None:
            Config_env.exp[args.env_surname]['start_with_gaussian_vel'] = True
            Config_env.exp[args.env_surname]['vel_noise'] = args.vel_noise

        test_phase = Test_phase(gpu=args.gpu,
                                env_name=args.env_name,
                                env_surname=args.env_surname,
                                n_steps=args.n_steps,
                                n_episodes=args.n_episodes,
                                max_episode_len=args.max_episode_len,
                                add_simu_time=args.add_simu_time,
                                t_simu=args.t_simu,
                                file_name=args.file_name.format(int(c), int(str(c).split('.')[1]),
                                                                args.transitions_gridsearch[transition]),
                                path_to_save=args.path_to_save,
                                dirname_to_save=args.dirname_to_save,
                                path_for_loading=args.path_for_loading,
                                dirname_for_loading=args.dirname_for_loading.format(int(c), int(str(c).split('.')[1])),
                                c=c,
                                init_state=args.init_state,
                                target_state=args.target_state,
                                save_obs=args.save_obs,
                                )
        scores = test_phase.main()
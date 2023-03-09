import argparse
import sys
from pfrl import experiments
from Classes import Test_phase


parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="results",help=("Directory path to save output files." " If it does not exist, it will be created."),)
parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32]")
parser.add_argument("--gpu", type=int, default=0, help="GPU name. -1 if no GPU.")
parser.add_argument("--env_name", type=str, default='env-3M-v0', help="environment name")
parser.add_argument("--env_surname", type=str, default='3M-v0', help="environment name")
parser.add_argument("--n_steps", type=int, default=None, help="total number of steps. If None, the number of episodes put will be taken into account.")
parser.add_argument("--n_episodes", type=int, default=1, help="Number of episodes to run.")
parser.add_argument("--max_episode_len", type=int, default=200, help="length of an episode")
parser.add_argument("--add_simu_time", type=bool, default=False, help="add time at the end of an episode, letting the system relax.")
parser.add_argument("--t_simu", type=int, default=None, help="length of the added time")
parser.add_argument("--file_name", type=str, default='c_{}_{}_transition_{}', help="file name for pos/vel/force results")
parser.add_argument("--path_to_save", type=str, default=None, help="path for pos/vel/force results")
parser.add_argument("--dirname_to_save", type=str, default=None, help="dirname to save pos/vel/force results")
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading models")
parser.add_argument("--dirname_for_loading", type=str, default=None, help="dirname for loading models")
parser.add_argument("--c", type=float, default=2.1, help="value of the dissipation coefficient")
parser.add_argument("--init_state", default=[[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
parser.add_argument("--target_state", default=[[0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]])
parser.add_argument("--transitions_gridsearch", default=['000_to_011', '111_to_011', '111_to_001', '000_to_010', '111_to_100', '000_to_101', '111_to_101', '000_to_110'])
parser.add_argument("--save_obs", type=bool, default=True, help="save pos/vel/force results")
args = parser.parse_args()

args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
print("Output files are saved in {}".format(args.outdir))

for transition in range(len(args.transitions_gridsearch)):

    test_phase = Test_phase(gpu=args.gpu,
                            env_name=args.env_name,
                            env_surname=args.env_surname,
                            n_steps=args.n_steps,
                            n_episodes=args.n_episodes,
                            max_episode_len=args.max_episode_len,
                            add_simu_time=args.add_simu_time,
                            t_simu=args.t_simu,
                            file_name=args.file_name.format(int(args.c), int(str(args.c).split('.')[1]), args.transitions_gridsearch[transition]),
                            path_to_save=args.path_to_save,
                            dirname_to_save=args.dirname_to_save,
                            path_for_loading=args.path_for_loading,
                            dirname_for_loading=args.dirname_for_loading,
                            c=args.c,
                            init_state=args.init_state[transition],
                            target_state=args.target_state[transition],
                            save_obs=args.save_obs,
                            )
    scores = test_phase.main()


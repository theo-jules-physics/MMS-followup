import argparse
import sys

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
parser.add_argument("--dirname", type=str, default='c_{}_{}', help="directory to put results in and to load models")
parser.add_argument("--tar_act_noise", type=float, default=0.5, help="maximum target actions noise")
parser.add_argument("--wrong_keys", type=bool, default=False, help="wrong_keys is True only if the loaded model does not have the same keys as the used model. Put False here.")
parser.add_argument("--c_gridsearch_overdamped", default=[2., 3., 4., 5., 6., 7., 8., 9., 10.], help="dissipation gridsearch toward the overdamped regime")
parser.add_argument("--c_gridsearch_inertial", default=[2., 1., 0.6, 0.3, 0.2, 0.1, 0.], help="dissipation gridsearch toward the inertial regime")
parser.add_argument("--toward_overdamped", type=bool, default=True, help="True if doing TL toward the overdamped regime. If false TL is done toward the inertial regime.")
args = parser.parse_args()


args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
print("Output files are saved in {}".format(args.outdir))

if args.toward_overdamped==True:
    c_gridsearch = args.c_gridsearch_overdamped
else:
    c_gridsearch = args.c_gridsearch_inertial

for dissipation in range(len(args.c_gridsearch)):

    Config_env.exp[args.env_surname]['c'] = args.c_gridsearch[dissipation]

    if dissipation==0:
        dirname_for_loading = None
    else:
        dirname_for_loading = args.dirname.format(int(args.c_gridsearch[dissipation-1]), int(str(args.c_gridsearch[dissipation-1]).split('.')[1]))

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
                  dirname_to_save=args.dirname.format(int(args.c_gridsearch[dissipation]), int(str(args.c_gridsearch[dissipation]).split('.')[1])),
                  dirname_for_loading=dirname_for_loading,
                  tar_act_noise=args.tar_act_noise,
                  wrong_keys=args.wrong_keys,
                  )

    agent, eval_env = train.main()


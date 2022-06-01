import argparse
import sys

from pfrl import experiments

from Classes import Train


parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="results",help=("Directory path to save output files." " If it does not exist, it will be created."),)
parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32]")
parser.add_argument("--put_seed", type=bool, default=True, help="put seeds for reproducibility")
parser.add_argument("--gpu", type=int, default=0, help="GPU name. -1 if no GPU.")
parser.add_argument("--env_name", type=str, default='env-3M-v0', help="environment name")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--rb_size", type=int, default=10**6, help="Replay buffer size")
parser.add_argument("--tau", type=float, default=5e-3, help="soft update tau")
parser.add_argument("--replay_start_size", type=int, default=10000, help="exploration time")
parser.add_argument("--minibatch_size", type=int, default=100, help="batchsize for updates")
parser.add_argument("--steps", type=int, default=8000000, help="total number of steps")
parser.add_argument("--eval_n_episodes", type=int, default=10, help="for tracking training")
parser.add_argument("--eval_interval", type=int, default=2000, help="for tracking training")
parser.add_argument("--train_max_episode_len", type=int, default=200, help="length of an episode")
parser.add_argument("--path_to_save", type=str, default=None, help="path for results and models")
parser.add_argument("--path_for_loading", type=str, default=None, help="path for loading models. If learning from scratch, put None.")
parser.add_argument("--dirname_to_save", type=str, default=None, help="dirname for saving results and models")
parser.add_argument("--dirname_for_loading", type=str, default=None, help="dirname for loading the model. If learning from scratch, put None.")
parser.add_argument("--tar_act_noise", type=float, default=0.5, help="maximum target actions noise")
parser.add_argument("--threshold", type=float, default=None, help="threshold on the difference of the force signal")
parser.add_argument("--noise", type=float, default=None, help="add noise to the observation. Do not put seeds if using it.")
parser.add_argument("--scheduler", default=None, help="scheduler torch.optim")
parser.add_argument("--automatically_stop", type=bool, default=False, help="automatically kills the code when a success rate is reached")
parser.add_argument("--success_threshold", type=float, default=0.95, help="success rate threshold for stopping learning")
parser.add_argument("--save_force_signal", type=bool, default=False, help="save force signal during training")
args = parser.parse_args()


args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
print("Output files are saved in {}".format(args.outdir))

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
              dirname_to_save=args.dirname_to_save,
              dirname_for_loading=args.dirname_for_loading,
              tar_act_noise=args.tar_act_noise,
              threshold=args.threshold,
              noise=args.noise,
              scheduler=args.scheduler,
              automatically_stop=args.automatically_stop,
              success_threshold=args.success_threshold,
              save_force_signal=args.save_force_signal
              )

agent, eval_env = train.main()
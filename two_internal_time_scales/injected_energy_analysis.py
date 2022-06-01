import argparse

from Classes import Injected_energy_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--path_for_loading", type=str, default=None, help="path for pos/vel/force results")
parser.add_argument("--dirname_for_loading", type=str, default=None)
parser.add_argument("--nb_of_tests", type=int, default=100)
parser.add_argument("--nb_ressort", type=int, default=3)
parser.add_argument("--transition_name", type=str, default='111_to_001', help="transition name for saving")
parser.add_argument("--c_gridsearch", default=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], help="dissipation gridsearch")
parser.add_argument("--path_for_fig", type=str, default=None, help="path for saving figure")
parser.add_argument("--fig_name", type=str, default=None, help="figure name")
args = parser.parse_args()


injected_energy = Injected_energy_analysis(path_for_loading=args.path_for_loading,
                                           dirname_for_loading=args.dirname_for_loading,
                                           nb_of_tests=args.nb_of_tests,
                                           c_gridsearch=args.c_gridsearch,
                                           nb_ressort=args.nb_ressort,
                                           transition_name=args.transition_name,
                                           path_for_fig=args.path_for_fig,
                                           fig_name=args.fig_name)

injected_energy.main()

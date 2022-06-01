# Multistable memory system

The aim of this project is to simulate a 1D multistable chain composed of coupled bistable spring-mass units and to control
it using a Reinforcement Learning agent, Twin-Delayed DDPG. We used the TD3 agent implemented here [PFRL](https://github.com/pfnet/pfrl).
This repository was used to produce the results of the article "Dynamically writing coupled memories using a reinforcement learning agent, meeting physical bounds" (https://arxiv.org/pdf/2205.03471.pdf).

### Description of the repository

The repertory [gym_systmemoire](gym_systmemoire) contains the environment simulating the multistable chain. 
The file [Config_env.py](Config_env.py) is used to configure the environment.
The repertory [pfrl-master](pfrl-master) contains the RL agent and functions to train it. This repertory can be found here [PFRL](https://github.com/pfnet/pfrl).
In this project, we have modified the file [train_agent.py](pfrl-master/pfrl/experiments/train_agent.py).
The file [Train_phase.py](Train_phase.py) is used to train the agent and the file [Test_phase.py](Test_phase.py) generates a chosen number of episodes or steps to test 
the learned models. 
The repertory [TL](TL) is used to do Transfer Learning from a regime to others by varying the friction coefficient (see fig. 2 b) of the article).
The repertory [two_internal_time_scales](two_internal_time_scales) is used to generate the data of fig. 3 and the repertory [scaling_analysis](scaling_analysis) is used to generate the 
data of fig. 4.
The repertory [func_plot](func_plot) contains scripts to plot the [learning dynamics](func_plot/plot_success_rate.py), the [force signal](func_plot/plot_force_signal.py)
and the [elongations of each mass](func_plot/plot_elongation.py).


### Installation

The environment can be installed using :

`python setup.py install`

PFRL can be installed using :

`cd pfrl-master`

`python setup.py install`

### Running

To train a new model :

`python3 Train_phase.py [options]`

If training a new model from an already pretrained model [c_2_0](results/c_2_0): 

`python3 Train_phase.py --path_for_loading "results/c_2_0"`

To test the model : 

`python3 Test_phase.py [options]`

To reproduce the results of Fig. 2 b : 

`cd TL`

`python3 generate_data.py --toward_overdamped True`

`python3 generate_data.py --toward_overdamped False`

`python3 plot_TL.py [options]`

To reproduce the results of Fig. 3 : 

`cd two_internal_time_scales`

`python3 generate_data.py [options]`

`python3 force_signal_analysis.py [options]`

`python3 injected_energy_analysis.py [options]`

To reproduce the results of Fig. 4 :

`cd scaling_analysis`

`python3 generate_data.py [options]`

`python3 plot_scaling.py [options]`

### License

[MIT License](LICENSE)

### DOI 

[![DOI](https://zenodo.org/badge/485346961.svg)](https://zenodo.org/badge/latestdoi/485346961)


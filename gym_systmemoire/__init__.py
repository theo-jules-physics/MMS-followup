from gym.envs.registration import register
import Config_env

register(
    id='env-3M',
    entry_point='gym_systmemoire.envs:Couplage_n_Env_Spring',
    max_episode_steps=200,
    kwargs=Config_env.exp['3M']
)

register(
    id='env-2M',
    entry_point='gym_systmemoire.envs:Couplage_n_Env_Spring',
    max_episode_steps=200,
    kwargs=Config_env.exp['2M']
)

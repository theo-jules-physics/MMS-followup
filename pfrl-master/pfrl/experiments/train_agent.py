import logging
import os

from pfrl.experiments.evaluator import Evaluator, save_agent
from pfrl.utils.ask_yes_no import ask_yes_no
import torch

import numpy as np

import sys
from Functions import get_success_rate


def save_agent_replay_buffer(agent, t, outdir, suffix="", logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, "{}{}.replay.pkl".format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info("Saved the current replay buffer to %s", filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=""):
    if hasattr(agent, "replay_buffer") and ask_yes_no(
        "Replay buffer has {} transitions. Do you save them to a file?".format(
            len(agent.replay_buffer)
        )
    ):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def train_agent(
    agent,
    env,
    steps,
    outdir,
    threshold,
    noise,
    scheduler=None,
    automatically_stop=False,
    success_threshold=0.9,
    save_force_signal=False,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    eval_during_episode=False,
    logger=None,
):

    logger = logger or logging.getLogger(__name__)
    episode_r = 0
    episode_idx = 0

    print('works with automatic shut down')

    # o_0, r_0
    obs = env.reset()

    print('reset : ', obs)

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    reward_history = []
    episode_len = 0

    actions = []
    actions.append(agent.act(obs))
    try:
        while t < steps:
            #add noise to observation, there must be no seeds
            if noise is not None:
                obs[:-env.nb_ressort] = obs[:-env.nb_ressort] + noise
            #to avoid abrupt changes in force
            if threshold is not None:
                action = min(max(agent.act(obs), actions[-1] - threshold), threshold + actions[-1])
            else:
                action = agent.act(obs)
            actions.append(action)
            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            for hook in step_hooks:
                hook(env, agent, t)

            episode_end = done or reset or t == steps
            if episode_end:
                logger.info(
                    "outdir:%s step:%s episode:%s R:%s",
                    outdir,
                    t,
                    episode_idx,
                    episode_r,
                )
                stats = agent.get_statistics()
                logger.info("statistics:%s", stats)
                reward_history.append(episode_r)
                episode_idx += 1

            if evaluator is not None and (episode_end or eval_during_episode):
                eval_score = evaluator.evaluate_if_necessary(t=t, episodes=episode_idx)
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                if (
                    successful_score is not None
                    and evaluator.max_score >= successful_score
                ):
                    break

            if episode_end:
                if t == steps:
                    break
                #save the number of steps in a file
                episode_len = np.array([episode_len])
                with open('{}/nb_of_steps.npy'.format(outdir), 'a') as f:
                    np.savetxt(f, episode_len)
                if save_force_signal==True:
                    with open('{}/force.npy'.format(outdir), 'a') as f:
                        np.savetxt(f, actions)
                if episode_idx > 100:
                    if automatically_stop==True:
                        success_rate = get_success_rate('{}/nb_of_steps.npy'.format(outdir), max_episode_len)
                        if success_rate>success_threshold:
                            sys.exit(0)
                #Scheduler
                if scheduler is not None:
                    scheduler.step()
                # Start a new episode
                episode_r = 0
                episode_len = 0
                obs = env.reset()
                actions = []
                actions.append(agent.act(obs))
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix="_finish")
    return eval_stats_history


def train_agent_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    threshold,
    noise,
    scheduler=None,
    automatically_stop=False,
    success_threshold=0.9,
    save_force_signal=False,
    checkpoint_freq=None,
    train_max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    eval_env=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    eval_during_episode=False,
    logger=None,
):
    """Train an agent while periodically evaluating it.

    Args:
        agent: A pfrl.agent.Agent
        env: Environment train the agent against.
        steps (int): Total number of timesteps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_episodes (int): Number of episodes at each evaluation phase.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output data.
        threshold : smooth force.
        noise : the agent does not have access to the true obs during training (it has only access to the true
        initial conditions).
        checkpoint_freq (int): frequency at which agents are stored.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        eval_during_episode (bool): Allow running evaluation during training episodes.
            This should be enabled only when `env` and `eval_env` are independent.
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent:
            raise ValueError(
                "{} does not support train_agent_with_evaluation().".format(hook)
            )

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        assert not eval_during_episode, (
            "To run evaluation during training episodes, you need to specify `eval_env`"
            " that is independent from `env`."
        )
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )

    eval_stats_history = train_agent(
        agent,
        env,
        steps,
        outdir,
        threshold,
        noise,
        scheduler,
        automatically_stop,
        success_threshold,
        save_force_signal,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        eval_during_episode=eval_during_episode,
        logger=logger,
    )

    return agent, eval_stats_history

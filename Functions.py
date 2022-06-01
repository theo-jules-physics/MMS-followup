import numpy as np

def get_success_rate(training_file_path, train_max_episode_len):
    nb_of_steps_train = np.loadtxt(training_file_path)[-100:]
    episode_success = nb_of_steps_train<train_max_episode_len
    nb_of_successes = 0.
    for i in range(len(episode_success)):
        if episode_success[i] == True:
            nb_of_successes += 1.
    success_rate = nb_of_successes/len(episode_success)
    return success_rate

def get_success_rate_2(nb_of_steps_train):
    episode_success = nb_of_steps_train < 200
    nb_of_successes = 0.
    for i in range(len(episode_success)):
        if episode_success[i] == True:
            nb_of_successes += 1.
    success_rate = nb_of_successes / len(episode_success)
    return success_rate

def get_stop(training_file_path, threshold):
    nb_of_steps_train = np.loadtxt(training_file_path)
    episode_stop = []
    for i in range(len(nb_of_steps_train) - 100):
        success_rate = get_success_rate_2(nb_of_steps_train[i:i+100])
        if success_rate > threshold:
            episode_stop.append(i+101)
    return episode_stop[0]
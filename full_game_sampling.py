import numpy as np
from CFR_external_sampling import traverse

def full_game_traversal(pid, net_arr, advantage_buffer_arr, strategy_buffer, t):

    for round_num in range(1, 6):
        round_vars = generate_initial_condition(round_num)
        traverse([], pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t)


def sample_n_monsters_removed(round_num):

    x = np.random.random()

    if round_num == 1:
        return 0

    elif round_num == 2:
        if (x < 0.6):
            return 0
        return 1

    elif round_num == 3:
        if (x < 0.25):
            return 0
        elif (x < 0.8):
            return 1
        return 2

    elif round_num == 4:
        if (x < 0.1):
            return 0
        elif (x < 0.45):
            return 1
        elif (x < 0.75):
            return 2
        return 3

    if (x < 0.07):
        return 0
    elif (x < 0.25):
        return 1
    elif (x < 0.60):
        return 2
    elif (x < 0.9):
        return 3
    return 4


def sample_n_artifacts_removed(round_num):

    x = np.random.random()

    if round_num == 1:
        return 0
    elif round_num == 2:
        if (x < 0.85):
            return 0
        return 1
    elif round_num == 3:
        if (x < 0.7):
            return 0
        elif (x < 0.95):
            return 1
        return 2
    elif round_num == 4:
        if (x < 0.45):
            return 1
        elif (x < 0.95):
            return 2
        return 3
    if (x < 0.15):
        return 0
    elif (x < 0.45):
        return 1
    elif (x < 0.75):
        return 2
    elif (x < 0.95):
        return 3
    return 4


def generate_removed_monsters_arr(round_num):

    n_removed_monsters = sample_n_monsters_removed(round_num)
    removed_monsters = [0] * 5

    if n_removed_monsters != 0:

        idx_list = list(np.random.choice(5, n_removed_monsters, replace=False))

        for idx in idx_list:
            removed_monsters[idx] = 1

    return removed_monsters


def generate_artifact_numbers(round_num):

    n_removed_artifacts = sample_n_artifacts_removed(round_num)

    if n_removed_artifacts != 0:

        n_destroyed_artifacts = np.random.choice(n_removed_artifacts)
        n_collected_artifacts = n_removed_artifacts - n_destroyed_artifacts

        return (n_destroyed_artifacts, n_collected_artifacts)

    return (0, 0)


def generate_initial_condition(round_num):

    round_vars = {'removed_monsters': [0] * 5,
            'n_collected_artifacts': 0,
            'n_destroyed_artifacts': 0,
            'round_num': round_num}

    round_vars = {}
    round_vars['round_num'] = round_num
    round_vars['removed_monsters'] = generate_removed_monsters_arr(round_num)

    (n_destroyed_artifacts, n_collected_artifacts) = generate_artifact_numbers(round_num)
    round_vars['n_destroyed_artifacts'] = n_destroyed_artifacts    
    round_vars['n_collected_artifacts'] = n_collected_artifacts

    return round_vars

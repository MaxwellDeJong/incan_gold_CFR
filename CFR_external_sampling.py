import numpy as np
import copy
import torch
from memory import Buffer

NUM_PLAYERS = 2


def calc_payoff(h, p, round_vars, update_round_vars=False):
    '''Calculates the payoff for player p from a history h.'''
    
    round_scores = np.zeros(NUM_PLAYERS)

    n_played_cards = 1 + int(len(h[1:]) / (NUM_PLAYERS + 1))
    num_artifact_pts = 0

    num_round_artifacts = 0
    num_tot_artifacts = round_vars['n_collected_artifacts']

    active_players = np.ones(NUM_PLAYERS)
    played_monsters = np.zeros(5)
    
    shared_gems = 0
    rem_gems = 0
    
    for i in range(n_played_cards):
        
        leaving_players = np.zeros(NUM_PLAYERS)

        history_idx = i * (NUM_PLAYERS + 1)
        drawn_card = h[history_idx]
        
        if len(h[history_idx+1:]) > NUM_PLAYERS:
            actions = h[history_idx+1:(i+1)*(NUM_PLAYERS+1)]
        else:
            n_missing_actions = NUM_PLAYERS - len(h[(history_idx+1):])
            actions = h[(history_idx+1):] + [0] * n_missing_actions
            
        if drawn_card[0] == 'M':
            monster_idx = int(drawn_card[1]) - 1
            played_monsters[monster_idx] += 1
        elif drawn_card[0] == 'A':
            num_tot_artifacts += 1
            num_round_artifacts += 1

            if (num_tot_artifacts > 3):
                num_artifact_pts += 10
            else:
                num_artifact_pts += 5
        else:
            gem_value = int(drawn_card)
            shared_gems += int(gem_value / np.sum(active_players))
            rem_gems += gem_value % np.sum(active_players)

        if np.any(played_monsters == 2):
            monster_idx = np.argwhere(played_monsters == 2)[0][0]

            if update_round_vars:
                round_vars['removed_monsters'][monster_idx] += 1
            return 0

        for j in range(len(actions)):
            action = actions[j]

            if (active_players[j] == 1) and (action == 0):
                
                leaving_players[j] = 1
                active_players[j] = 0
                
        n_leaving_players = np.sum(leaving_players)
        
        if n_leaving_players == 0:
            rem_gem_contrib = 0
        else:
            rem_gem_contrib = int(rem_gems / n_leaving_players)

        rem_gems -= (rem_gem_contrib * n_leaving_players)

        for j in range(NUM_PLAYERS):
            if (leaving_players[j] == 1):
                if n_leaving_players == 1:
                    round_scores[j] += num_artifact_pts
                    if update_round_vars:
                        round_vars['n_collected_artifacts'] += num_round_artifacts
                    num_artifact_pts = 0
                    num_round_artifacts = 0

                round_scores[j] += shared_gems
                round_scores[j] += rem_gem_contrib

        if leaving_players[p] == 1:
            if update_round_vars:
                round_vars['n_destroyed_artifacts'] += num_round_artifacts
            return round_scores[p]
                
    if update_round_vars:
        round_vars['n_destroyed_artifacts'] += num_round_artifacts
    return round_scores[p]


def double_monster(h):
    '''Checks if a second monster has been drawn for history h.'''
    
    n_played_cards = 1 + int(len(h[1:]) / (NUM_PLAYERS + 1))
    
    if n_played_cards < 2:
        return False

    n_monsters_played = np.zeros(5)

    for i in range(n_played_cards):
        card_idx = i * (NUM_PLAYERS + 1)

        if h[card_idx][0] == 'M':
            monster_idx = int(h[card_idx][1]) - 1

            n_monsters_played[monster_idx] += 1

    double_monster_termination = np.any(n_monsters_played > 1)

    return double_monster_termination


def all_leave(h):
    '''Checks if all players have left the temple.'''
    
    leave_termination = False
    
    if len(h) < (NUM_PLAYERS + 1):
        return False
    
    prev_actions = h[-1 * (NUM_PLAYERS + 1):]
    
    player_actions = [prev_actions[i] for i in range(NUM_PLAYERS + 1) if not is_str(prev_actions[i])] 
    
    return (np.sum(player_actions) == 0)


def is_hist_terminal(h):
    '''Checks if a history is terminal.'''
    
    double_monster_termination = double_monster(h)
    all_leave_termination = all_leave(h)
    
    return double_monster_termination or all_leave_termination


def is_hist_p_terminal(h, p):
    '''Checks if a history is terminal for player p.'''
    
    double_monster_termination = double_monster(h)
    if len(h) < NUM_PLAYERS+1:
        p_termination = False
    else:
        p_termination = (h[-1 * (NUM_PLAYERS+1)] == 0)
    
    return double_monster_termination or p_termination


def available_actions(h, player):
    
    if len(h) < (NUM_PLAYERS + 1):
        return [0, 1]
    
    prev_action = h[-1 * (NUM_PLAYERS + 1)]
    
    if (prev_action == 0):
        return [0]
    
    return [0, 1]


def is_chance_node(h):
    
    if len(h) == 0:
        return True
    
    if len(h) < (NUM_PLAYERS + 1):
        return False
    
    return is_str(h[-1 * (NUM_PLAYERS + 1)])


def calc_n_max_artifact_pts(round_vars):

    rd = round_vars['round_num']
    n_collected_artifacts = round_vars['n_collected_artifacts']
    n_destroyed_artifacts = round_vars['n_destroyed_artifacts']

    max_n_artifacts = rd - (n_collected_artifacts + n_destroyed_artifacts)

    artifact_pts = 0
    curr_n_collected_artifacts = n_collected_artifacts

    for art in range(max_n_artifacts):
        curr_n_collected_artifacts += 1

        if (curr_n_collected_artifacts > 3):
            artifact_pts += 10
        else:
            artifact_pts += 5

    return artifact_pts


def embed_history(h, round_vars):
    
    embedded_history = {'n_gem_cards_played': 0,
                        'n_shared_gems': 0,
                        'n_rem_gems': 0,
                        'n_artifact_pts': 0,
                        'n_full_monsters': 0,
                        'n_handicapped_monsters': 0,
                        'n_rem_players': NUM_PLAYERS,
                        'n_removed_monsters': 0,
                        'n_max_artifact_pts': 5}

    embedded_history['n_removed_monsters'] = np.sum(round_vars['removed_monsters'])
    embedded_history['n_max_artifact_pts'] = calc_n_max_artifact_pts(round_vars)

    if len(h) == 0:
        return embedded_history
    
    round_scores = np.zeros(NUM_PLAYERS)

    n_played_cards = 1 + int(len(h[1:]) / (NUM_PLAYERS + 1))
    n_collected_artifacts = 0
    
    active_players = np.ones(NUM_PLAYERS)
    
    n_monsters_played = np.zeros(5)

    for i in range(n_played_cards):

        leaving_players = np.zeros(NUM_PLAYERS)
        
        history_idx = i * (NUM_PLAYERS + 1)
        drawn_card = h[history_idx]
        
        if len(h[history_idx+1:]) > NUM_PLAYERS:
            actions = h[history_idx+1:(i+1)*(NUM_PLAYERS+1)]
        else:
            actions = h[history_idx+1:]

        if drawn_card[0] == 'M':
            monster_idx = int(drawn_card[1]) - 1

            if round_vars['removed_monsters'][monster_idx] == 0:
                embedded_history['n_full_monsters'] += 1
            else:
                embedded_history['n_handicapped_monsters'] += 1
            
            n_monsters_played[monster_idx] += 1
            
        elif drawn_card[0] == 'A':
            if (n_collected_artifacts + round_vars['n_collected_artifacts']) > 3:
                embedded_history['n_artifact_pts'] += 10
            else:
                embedded_history['n_artifact_pts'] += 5

            n_collected_artifacts += 1

        else:
            embedded_history['n_gem_cards_played'] += 1
            gem_value = int(drawn_card)
            
            embedded_history['n_shared_gems'] += int(gem_value / np.sum(active_players))
            embedded_history['n_rem_gems'] += gem_value % np.sum(active_players)
            
        # Only distribute remaining gems and artifacts if all players
        # have acted
        if len(actions) < NUM_PLAYERS:
            break

        for j in range(len(actions)):
            action = actions[j]

            if (active_players[j] == 1) and (action == 0):

                leaving_players[j] = 1
                active_players[j] = 0
                    
        n_leaving_players = np.sum(leaving_players)
        
        if n_leaving_players == 0:
            rem_gem_contrib = 0
        else:
            rem_gem_contrib = int(embedded_history['n_rem_gems'] / n_leaving_players)

        embedded_history['n_rem_gems'] -= (rem_gem_contrib * n_leaving_players)

        for j in range(len(actions)):
            if (leaving_players[j] == 1) and (n_leaving_players == 1):
                if embedded_history['n_artifact_pts'] != 0:
                    embedded_history['n_artifact_pts'] = 0
                    embedded_history['n_max_artifact_pts'] = 0
                        
    embedded_history['n_rem_players'] = np.sum(active_players)

    return embedded_history


def predict_strategy(net_arr, pid, embedded_history):

    embedded_state = list(embedded_history.values())



def full_game_traversal(pid, net_arr, advantage_buffer_arr, strategy_buffer, t):

    # Keep track of number of removed artifacts, number of removed gems, and round
    round_vars = {'removed_monsters': [0] * 5,
            'n_collected_artifacts': 0,
            'n_destroyed_artifacts': 0,
            'round_num': 1}

    for rd in range(1, 6):
        h = []
        round_vars['round_num'] = rd
        traverse(h, pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t)


def traverse(h, pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t):

    if is_hist_terminal(h):
        return calc_payoff(h, pid, round_vars)
    
    # Chance node
    if is_chance_node(h):
        chance_node = ChanceNode(h, round_vars)
        action = chance_node.sample_action()
        
        return traverse(h + [action], pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t)
    
    embedded_history = embed_history(h, round_vars)
    
    if (get_active_player(h) == pid):

        if is_hist_p_terminal(h, pid):
            return calc_payoff(h, pid, round_vars)
        
        strategy = net_arr.get_strategy(embedded_history, pid)

        regret_arr = np.zeros(2)
        
        for a in [0, 1]:
            regret_arr[a] = traverse(h + [a], pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t)

        mean_value = np.dot(strategy, regret_arr)
        regret_arr -= mean_value
        advantage_buffer_arr[pid].add(embedded_history, t, regret_arr)

        return mean_value
        
    else:
        
        opp_idx = get_active_player(h)
        avail_actions = available_actions(h, opp_idx)

        if len(avail_actions) == 1:
            return traverse(h + [0], pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t)
        
        strategy = net_arr.get_strategy(embedded_history, opp_idx)
        strategy_buffer.add(embedded_history, t, strategy)
        action = np.random.choice([0, 1], p=strategy)

        return traverse(h + [action], pid, round_vars, net_arr, advantage_buffer_arr, strategy_buffer, t)
   
    
def is_str(elem):
    
    py_str = type(elem) is str
    np_str = type(elem) is np.str_
    
    return (py_str or np_str)
    
    
def get_active_player(h):
    
    player = 0
    
    if len(h) < NUM_PLAYERS+1:
        return len(h) - 1
    
    for i in range(NUM_PLAYERS+1):
        if is_str(h[-1 + i]):
            return player
        else:
            player += 1
            
    return player
       
        
class ChanceNode():
    
    def __init__(self, h, round_vars):

        self.rem_cards, self.probs = self.get_remaining_cards(h, round_vars)
        
        
    def get_all_cards(self, round_vars):
        
        all_gems = ['1', '2', '3', '4', '5', '7', '9', '11', '13', '14', '15']
        gem_counts = [1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1]

        n_artifacts = round_vars['round_num'] - (round_vars['n_collected_artifacts'] + round_vars['n_destroyed_artifacts'])

        all_artifacts = []
        for i in range(1, n_artifacts + 1):
            all_artifacts.append('A' + str(i))

        artifact_counts = len(all_artifacts) * [1]
            
        all_monsters = ['M' + str(num) for num in range(1, 6)]
        monster_counts = [3 for _ in range(1, 6)]

        for monster_id in range(5):
            monster_counts[monster_id] -= round_vars['removed_monsters'][monster_id] 
                    
        all_cards = all_gems + all_artifacts + all_monsters
        all_counts = gem_counts + artifact_counts + monster_counts
            
        return (all_cards, all_counts)

       
    def get_remaining_cards(self, actions_history, round_vars):
            
        (all_cards, all_counts) = self.get_all_cards(round_vars)
            
        if len(actions_history) == 0:
            probs = all_counts / np.sum(all_counts)
            return (all_cards, probs)
            
        n_played_cards = 1 + int(len(actions_history[1:]) / (NUM_PLAYERS + 1))
        
        idx_to_remove = []

        for i in range(n_played_cards):
            
            played_card = actions_history[i * (NUM_PLAYERS + 1)]
                
            master_idx = all_cards.index(played_card)
                
            all_counts[master_idx] -= 1
                
            if (all_counts[master_idx] == 0):
                idx_to_remove.append(master_idx)
                
        rem_cards = [all_cards[i] for i in range(len(all_cards)) if i not in idx_to_remove]
        rem_counts = [all_counts[i] for i in range(len(all_counts)) if i not in idx_to_remove]
        
        n_cards = np.sum(rem_counts)
        probs = rem_counts / n_cards
                
        return (rem_cards, probs)
        
        
    def sample_action(self):
        
        return np.random.choice(self.rem_cards, p=self.probs)


if __name__ == '__main__':

    h = ['M3', 1, 1, '3', 1, 1, 'A1', 1, 1, 'M3']

    round_vars = {'removed_monsters': [0] * 5,
            'n_collected_artifacts': 1,
            'n_destroyed_artifacts': 0,
            'round': 2}

    round_vars['n_collected_artifacts'] = 1
    round_vars['removed_monsters'][2] = 1

    n_artifacts = round_vars['round'] - (round_vars['n_collected_artifacts'] + round_vars['n_destroyed_artifacts'])
    
    print(n_artifacts)

    #CN = ChanceNode(h, round_vars)

    #all_cards = CN.get_all_cards(round_vars)
    #rem_cards = CN.get_remaining_cards(h, round_vars)

    #print('All cards: ', all_cards)
    #print('Rem cards: ', rem_cards)

import torch
import numpy as np
import copy
from deep_cfr_net import AdvantageNet

class StrategyNetworks():

    def __init__(self, n_players=2, net_arr=[]):

        if net_arr == []:

            self.net_arr = [None] * n_players

            for i in range(n_players):
                self.net_arr[i] = AdvantageNet()

        else:
            self.net_arr = net_arr


    def get_strategy(self, embedded_history, p):

        with torch.no_grad():
            x = torch.tensor(list(embedded_history.values())).float()
            values = self.net_arr[p].forward(x).numpy()

        strategy = self.regret_min(values)
        return strategy


    def regret_min(self, values):

        regret_arr = np.clip(values, a_min=0, a_max=None)

        regret_sum = np.sum(regret_arr)

        if regret_sum < 1e-3:
            return np.ones(len(self.net_arr)) / (len(self.net_arr))

        return regret_arr / regret_sum


    def deepcopy(self):

        copy_net = [None] * len(self.net_arr)

        for i in range(len(copy_net)):
            copy_net[i] = copy.deepcopy(self.net_arr[i])

        new_strategy_net = StrategyNetworks(len(copy_net), copy_net)

        return new_strategy_net
